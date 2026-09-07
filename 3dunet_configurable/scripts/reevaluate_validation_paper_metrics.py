#!/usr/bin/env python3
import argparse
import csv
import logging
import shlex
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ProteinLigandDatasetWithH5
from main import as_list, build_transforms, create_model, load_config, resolve_dataset_lists
from utils.pocket_metrics import (
    POCKET_PER_PROTEIN_FIELDNAMES,
    POCKET_SUMMARY_FIELDNAMES,
    POCKET_TOPK_COMPONENT_FIELDNAMES,
    POCKET_TOPK_PER_PROTEIN_FIELDNAMES,
    POCKET_TOPK_SUMMARY_FIELDNAMES,
    evaluate_pocket_metrics_for_sample,
    evaluate_topk_metrics_for_sample,
    normalize_postprocess_mode,
    resolve_selection_score_config,
    select_best_paper_summary,
    summarize_pocket_metrics,
    summarize_topk_pocket_metrics,
)
from utils.training import (
    calculate_binary_stats_from_counts,
    calculate_binary_stats_from_probs,
    get_device,
    set_reproducibility,
)


PAPER_CONTEXT_FIELDNAMES = [
    "run_root",
    "run",
    "config",
    "checkpoint",
    "model",
    "feature_set_name",
    "feature_name",
    "fold",
    "postprocess_mode",
]
TOPK_EXTRA_FIELDNAMES = ["run_root", "config", "fold"]
RUN_INVENTORY_FIELDNAMES = [
    "run_root",
    "run",
    "config",
    "checkpoint",
    "model",
    "base_features",
    "feature_set_name",
    "feature_name",
    "fold",
    "validation_cases",
    "status",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate completed validation checkpoints with current paper metrics "
            "(DCC/DCA/DVO/PLI) without retraining."
        )
    )
    parser.add_argument(
        "--runs-root",
        action="append",
        required=True,
        help="Run root containing run_name/config_snapshot.yml folders. Repeatable.",
    )
    parser.add_argument(
        "--run-name-file",
        default="",
        help="Optional file containing one run directory name per line; limits re-evaluation to those runs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-name",
        default="best_model_in_terms_of_validation_paper_f1",
        help="Checkpoint filename fragment to evaluate.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="Override validation batch size.")
    parser.add_argument("--base-features", type=int, default=0, help="Override base feature count.")
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument(
        "--postprocess-modes",
        default="",
        help="Comma-separated modes. Default: config primary postprocess plus comparison_postprocess.",
    )
    parser.add_argument("--dcc-reference", default="label_center")
    parser.add_argument("--top-k", default="1,2,3")
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated thresholds to evaluate. Default: config threshold_sweep plus fixed threshold.",
    )
    parser.add_argument("--disable-topk", action="store_true")
    parser.add_argument("--reference-pocket-count", type=int, default=1)
    parser.add_argument("--no-top-n-plus-2", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue with remaining runs if one fails.")
    return parser.parse_args()


def parse_csv_items(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_run_name_filter(path_value):
    if not path_value:
        return None
    names = set()
    path = Path(path_value)
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.update(parse_csv_items(line))
    return names


def choose_device(requested):
    if requested == "auto":
        return get_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available.")
    return requested


def write_header(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path, fieldnames, rows):
    if not rows:
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def parse_model_from_checkpoint(checkpoint_path):
    name = checkpoint_path.name
    for marker in (
        "_best_model_in_terms_of_validation_paper_f1.pth",
        "_best_model_in_terms_of_training_score.pth",
        "_final_model.pth",
    ):
        if name.endswith(marker):
            return name[: -len(marker)]
    return name.split("_", 1)[0]


def parse_base_features(run_dir, default=8):
    command_path = run_dir / "run_command.txt"
    if not command_path.exists():
        return default
    try:
        parts = shlex.split(command_path.read_text().strip())
    except ValueError:
        return default
    for idx, part in enumerate(parts):
        if part == "--base_features" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return default
    return default


def clean_state_dict(state_dict, model_state_dict):
    has_module = any(key.startswith("module.") for key in state_dict)
    model_has_module = any(key.startswith("module.") for key in model_state_dict)
    if has_module and not model_has_module:
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    if model_has_module and not has_module:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return state_dict


def discover_runs(runs_roots, checkpoint_fragment, base_features_override=0):
    runs = []
    for root_value in runs_roots:
        runs_root = Path(root_value)
        for config_path in sorted(runs_root.rglob("config_snapshot.yml")):
            run_dir = config_path.parent
            weights_dir = run_dir / "weights"
            if not weights_dir.exists():
                continue
            candidates = sorted(weights_dir.glob(f"*{checkpoint_fragment}*.pth"))
            if not candidates:
                continue
            checkpoint = candidates[0]
            model = parse_model_from_checkpoint(checkpoint)
            config = load_config(str(config_path))
            feature_set = config.get("feature_set", {})
            runs.append(
                {
                    "run_root": runs_root,
                    "run_dir": run_dir,
                    "run": run_dir.name,
                    "config": config_path,
                    "checkpoint": checkpoint,
                    "model": model,
                    "base_features": base_features_override or parse_base_features(run_dir),
                    "feature_set_name": feature_set.get("name", ""),
                    "feature_name": feature_set.get("feature_name", feature_set.get("name", "")),
                    "fold": feature_set.get("fold", ""),
                }
            )
    return runs


def make_model(run, config, device):
    logger = logging.getLogger("reevaluate_validation_paper_metrics")
    model = create_model(
        model_class=run["model"],
        in_channels=len(config["features"]),
        base_features=run["base_features"],
        model_dropout=config.get("model", {}).get("dropout", 0.5),
        device=device,
        logger=logger,
    )
    state = torch.load(run["checkpoint"], map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(clean_state_dict(state, model.state_dict()))
    model.eval()
    return model


def resolve_postprocess_modes(paper_config, override):
    if override:
        return [normalize_postprocess_mode(mode) for mode in parse_csv_items(override)]
    modes = [normalize_postprocess_mode(paper_config.get("postprocess", "raw"))]
    for mode in as_list(paper_config.get("comparison_postprocess", ["kalasanty_puresnet"])):
        normalized = normalize_postprocess_mode(mode)
        if normalized not in modes:
            modes.append(normalized)
    return modes


def evaluate_run(run, args, device):
    config = resolve_dataset_lists(load_config(str(run["config"])), str(run["config"]))
    set_reproducibility(int(config.get("seed", config.get("training", {}).get("seed", 42))))
    validation_config = config.get("validation", {})
    paper_config = validation_config.get("paper_metrics", {})

    batch_size = args.batch_size or int(validation_config.get("batch_size", 8))
    dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"].get("validation"),
        transform=build_transforms(config, training=False),
        config_path=str(run["config"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    model = make_model(run, config, device)

    if args.thresholds:
        thresholds = sorted({float(value) for value in parse_csv_items(args.thresholds)})
    else:
        fixed_threshold = float(validation_config.get("threshold", 0.5))
        threshold_sweep = validation_config.get(
            "threshold_sweep",
            [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        thresholds = sorted({float(value) for value in [*threshold_sweep, fixed_threshold]})
    postprocess_modes = resolve_postprocess_modes(paper_config, args.postprocess_modes)
    top_k_values = tuple(sorted({int(item) for item in parse_csv_items(args.top_k)}))

    dcc_cutoff = float(paper_config.get("dcc_cutoff_angstrom", 4.0))
    dca_cutoff = float(paper_config.get("dca_cutoff_angstrom", 4.0))
    min_component_voxels = int(paper_config.get("min_component_voxels", 5))
    min_component_volume = paper_config.get("min_component_volume_angstrom3", 50.0)
    selection_config = resolve_selection_score_config(paper_config)

    context = {
        "run_root": str(run["run_root"]),
        "run": run["run"],
        "config": str(run["config"]),
        "checkpoint": run["checkpoint"].name,
        "model": run["model"],
        "feature_set_name": run["feature_set_name"],
        "feature_name": run["feature_name"],
        "fold": run["fold"],
    }
    sweep_counts = {threshold: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for threshold in thresholds}
    paper_rows_by_postprocess = {mode: [] for mode in postprocess_modes}
    topk_rows_by_postprocess = {mode: [] for mode in postprocess_modes}
    topk_component_rows_by_postprocess = {mode: [] for mode in postprocess_modes}

    sample_offset = 0
    feature_names = list(getattr(dataset, "feature_names", []))
    with torch.no_grad():
        for batch_idx, (protein, label) in enumerate(loader, start=1):
            feature_batch = protein.detach().cpu().numpy()
            protein = protein.to(device)
            probabilities = torch.sigmoid(model(protein).squeeze(1)).detach().cpu().numpy()
            targets = label.detach().cpu().numpy()

            for threshold in thresholds:
                batch_stats = calculate_binary_stats_from_probs(targets, probabilities, threshold)
                for key in sweep_counts[threshold]:
                    sweep_counts[threshold][key] += batch_stats[key]

            for sample_idx in range(probabilities.shape[0]):
                dataset_idx = sample_offset + sample_idx
                protein_name = dataset.samples[dataset_idx][0]
                metadata = dataset.get_metadata(dataset_idx)
                ligand_mask = dataset.load_metric_mask(dataset_idx, "features", "ligand")
                for postprocess_mode in postprocess_modes:
                    mode_min_component_volume = (
                        float(min_component_volume)
                        if min_component_volume is not None
                        and postprocess_mode not in {"raw", "custom_rank"}
                        else None
                    )
                    paper_rows = evaluate_pocket_metrics_for_sample(
                        probabilities=probabilities[sample_idx],
                        label_mask=targets[sample_idx],
                        ligand_mask=ligand_mask,
                        protein_name=protein_name,
                        thresholds=thresholds,
                        resolution=metadata["resolution"],
                        max_distance_angstrom=metadata["max_distance_angstrom"],
                        dcc_cutoff_angstrom=dcc_cutoff,
                        dca_cutoff_angstrom=dca_cutoff,
                        min_component_voxels=min_component_voxels,
                        min_component_volume_angstrom3=mode_min_component_volume,
                        postprocess_mode=postprocess_mode,
                        top_k_values=top_k_values,
                        dcc_reference=args.dcc_reference,
                        feature_volume=feature_batch[sample_idx],
                        feature_names=feature_names,
                    )
                    paper_rows_by_postprocess[postprocess_mode].extend(
                        {**context, "postprocess_mode": postprocess_mode, "epoch": "reeval", **row}
                        for row in paper_rows
                    )

                    if not args.disable_topk:
                        component_rows, topk_rows = evaluate_topk_metrics_for_sample(
                            probabilities=probabilities[sample_idx],
                            label_mask=targets[sample_idx],
                            ligand_mask=ligand_mask,
                            protein_name=protein_name,
                            thresholds=thresholds,
                            resolution=metadata["resolution"],
                            max_distance_angstrom=metadata["max_distance_angstrom"],
                            dcc_cutoff_angstrom=dcc_cutoff,
                            dca_cutoff_angstrom=dca_cutoff,
                            min_component_voxels=min_component_voxels,
                            min_component_volume_angstrom3=mode_min_component_volume,
                            postprocess_mode=postprocess_mode,
                            top_k_values=top_k_values,
                            reference_pocket_count=args.reference_pocket_count,
                            include_top_n_plus_2=not args.no_top_n_plus_2,
                            dcc_reference=args.dcc_reference,
                            feature_volume=feature_batch[sample_idx],
                            feature_names=feature_names,
                        )
                        topk_context = {
                            "run_root": str(run["run_root"]),
                            "config": str(run["config"]),
                            "fold": run["fold"],
                            "epoch": "reeval",
                            "run": run["run"],
                            "model": run["model"],
                            "feature_family": run["feature_name"],
                            "apbs_variant": "",
                            "checkpoint": run["checkpoint"].name,
                            "postprocess_mode": postprocess_mode,
                        }
                        topk_component_rows_by_postprocess[postprocess_mode].extend(
                            {**topk_context, **row} for row in component_rows
                        )
                        topk_rows_by_postprocess[postprocess_mode].extend(
                            {**topk_context, **row} for row in topk_rows
                        )

            sample_offset += probabilities.shape[0]
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
                logging.info(
                    "%s: validation batch %d/%d",
                    run["run"],
                    batch_idx,
                    len(loader),
                )

    voxel_summary_by_threshold = {
        threshold: calculate_binary_stats_from_counts(
            counts["tp"], counts["fp"], counts["tn"], counts["fn"], threshold
        )
        for threshold, counts in sweep_counts.items()
    }

    paper_summary_rows = []
    paper_best_rows = []
    paper_per_protein_rows = []
    topk_per_protein_rows = []
    topk_component_rows = []
    topk_summary_rows = []

    for postprocess_mode in postprocess_modes:
        summaries = summarize_pocket_metrics(
            paper_rows_by_postprocess[postprocess_mode],
            thresholds,
            selection_metric=selection_config["selection_metric"],
            selection_dvo_weight=selection_config["selection_dvo_weight"],
            selection_pli_weight=selection_config["selection_pli_weight"],
            selection_voxel_f1_weight=selection_config["selection_voxel_f1_weight"],
            selection_dca_weight=selection_config["selection_dca_weight"],
            selection_no_dcc_score_scale=selection_config["selection_no_dcc_score_scale"],
            selection_max_mean_predicted_positive_voxels=selection_config[
                "selection_max_mean_predicted_positive_voxels"
            ],
            selection_weights=selection_config["selection_weights"],
            selection_no_prediction_weight=selection_config["selection_no_prediction_weight"],
            selection_volume_penalty_power=selection_config["selection_volume_penalty_power"],
            selection_min_paper_f1=selection_config["selection_min_paper_f1"],
            selection_below_min_paper_f1_score_scale=selection_config[
                "selection_below_min_paper_f1_score_scale"
            ],
            voxel_summary_by_threshold=voxel_summary_by_threshold,
        )
        summaries = [{**context, "postprocess_mode": postprocess_mode, "epoch": "reeval", **row} for row in summaries]
        best = select_best_paper_summary(summaries)
        paper_summary_rows.extend(summaries)
        if best:
            paper_best_rows.append(best)
        paper_per_protein_rows.extend(paper_rows_by_postprocess[postprocess_mode])

        if not args.disable_topk:
            topk_rows = topk_rows_by_postprocess[postprocess_mode]
            topk_component_rows.extend(topk_component_rows_by_postprocess[postprocess_mode])
            topk_per_protein_rows.extend(topk_rows)
            topk_summary_rows.extend(summarize_topk_pocket_metrics(topk_rows))

    return {
        "inventory": {
            **context,
            "base_features": run["base_features"],
            "validation_cases": len(dataset),
            "status": "ok",
            "error": "",
        },
        "paper_summary_rows": paper_summary_rows,
        "paper_best_rows": paper_best_rows,
        "paper_per_protein_rows": paper_per_protein_rows,
        "topk_summary_rows": topk_summary_rows,
        "topk_per_protein_rows": topk_per_protein_rows,
        "topk_component_rows": topk_component_rows,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_summary_path = output_dir / "paper_metrics_by_threshold.csv"
    paper_best_path = output_dir / "paper_metrics_best.csv"
    paper_per_protein_path = output_dir / "paper_metrics_per_protein.csv"
    topk_summary_path = output_dir / "topk_metrics_by_threshold.csv"
    topk_per_protein_path = output_dir / "topk_metrics_per_protein.csv"
    topk_component_path = output_dir / "topk_component_metrics.csv"
    inventory_path = output_dir / "run_inventory.csv"

    write_header(inventory_path, RUN_INVENTORY_FIELDNAMES)
    write_header(paper_summary_path, PAPER_CONTEXT_FIELDNAMES + POCKET_SUMMARY_FIELDNAMES)
    write_header(paper_best_path, PAPER_CONTEXT_FIELDNAMES + POCKET_SUMMARY_FIELDNAMES)
    write_header(paper_per_protein_path, PAPER_CONTEXT_FIELDNAMES + POCKET_PER_PROTEIN_FIELDNAMES)
    if not args.disable_topk:
        write_header(topk_summary_path, TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_SUMMARY_FIELDNAMES)
        write_header(topk_per_protein_path, TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_PER_PROTEIN_FIELDNAMES)
        write_header(topk_component_path, TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_COMPONENT_FIELDNAMES)

    runs = discover_runs(args.runs_root, args.checkpoint_name, args.base_features)
    run_name_filter = load_run_name_filter(args.run_name_file)
    if run_name_filter:
        discovered_count = len(runs)
        runs = [run for run in runs if run["run"] in run_name_filter]
        missing = sorted(run_name_filter.difference({run["run"] for run in runs}))
        logging.info(
            "Run-name filter: %d/%d discovered runs selected",
            len(runs),
            discovered_count,
        )
        if missing:
            logging.warning("Missing requested runs: %s", ", ".join(missing[:20]))
    if args.limit_runs:
        runs = runs[: args.limit_runs]
    if not runs:
        raise SystemExit("No runs found with requested checkpoint fragment.")

    logging.info("Device: %s", device)
    logging.info("Runs: %d", len(runs))
    for idx, run in enumerate(runs, start=1):
        logging.info("[%d/%d] %s", idx, len(runs), run["run"])
        try:
            result = evaluate_run(run, args, device)
            append_rows(inventory_path, RUN_INVENTORY_FIELDNAMES, [result["inventory"]])
            append_rows(
                paper_summary_path,
                PAPER_CONTEXT_FIELDNAMES + POCKET_SUMMARY_FIELDNAMES,
                result["paper_summary_rows"],
            )
            append_rows(
                paper_best_path,
                PAPER_CONTEXT_FIELDNAMES + POCKET_SUMMARY_FIELDNAMES,
                result["paper_best_rows"],
            )
            append_rows(
                paper_per_protein_path,
                PAPER_CONTEXT_FIELDNAMES + POCKET_PER_PROTEIN_FIELDNAMES,
                result["paper_per_protein_rows"],
            )
            if not args.disable_topk:
                append_rows(
                    topk_summary_path,
                    TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_SUMMARY_FIELDNAMES,
                    result["topk_summary_rows"],
                )
                append_rows(
                    topk_per_protein_path,
                    TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_PER_PROTEIN_FIELDNAMES,
                    result["topk_per_protein_rows"],
                )
                append_rows(
                    topk_component_path,
                    TOPK_EXTRA_FIELDNAMES + POCKET_TOPK_COMPONENT_FIELDNAMES,
                    result["topk_component_rows"],
                )
        except Exception as exc:
            logging.exception("Failed: %s", run["run"])
            append_rows(
                inventory_path,
                RUN_INVENTORY_FIELDNAMES,
                [
                    {
                        "run_root": str(run["run_root"]),
                        "run": run["run"],
                        "config": str(run["config"]),
                        "checkpoint": str(run["checkpoint"]),
                        "model": run["model"],
                        "base_features": run["base_features"],
                        "feature_set_name": run["feature_set_name"],
                        "feature_name": run["feature_name"],
                        "fold": run["fold"],
                        "validation_cases": "",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
            )
            if not args.keep_going:
                raise

    logging.info("Wrote: %s", output_dir)


if __name__ == "__main__":
    main()
