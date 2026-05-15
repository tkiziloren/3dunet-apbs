#!/usr/bin/env python3
import argparse
import csv
import logging
import re
import shlex
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ProteinLigandDatasetWithH5
from main import build_transforms, create_model, load_config, resolve_dataset_lists
from utils.pocket_metrics import (
    POCKET_TOPK_COMPONENT_FIELDNAMES,
    POCKET_TOPK_PER_PROTEIN_FIELDNAMES,
    POCKET_TOPK_SUMMARY_FIELDNAMES,
    evaluate_topk_metrics_for_sample,
    normalize_postprocess_mode,
    summarize_topk_pocket_metrics,
)
from utils.training import get_device, set_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate completed Work8 checkpoints with component-level Top-k pocket metrics."
    )
    parser.add_argument(
        "--runs-root",
        default="/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040",
        help="Root directory that contains completed Work8 run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/work8a_topk_metrics_2026-05-15",
        help="Directory where Work8A CSV/Markdown outputs will be written.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_model_in_terms_of_validation_paper_f1",
        help="Checkpoint filename fragment to evaluate.",
    )
    parser.add_argument("--top-k", default="1,2,3", help="Comma-separated Top-k values to report.")
    parser.add_argument(
        "--reference-pocket-count",
        type=int,
        default=1,
        help="Known reference pocket count per case. For current scPDB cache this is one.",
    )
    parser.add_argument(
        "--no-top-n-plus-2",
        action="store_true",
        help="Disable the Top-(n+2) protocol row.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=0,
        help="Debug helper: evaluate only the first N discovered runs.",
    )
    return parser.parse_args()


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested):
    if requested == "auto":
        return get_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available.")
    return requested


def parse_model_from_checkpoint(checkpoint_path):
    name = checkpoint_path.name
    marker = "_best_model_in_terms_of_validation_paper_f1.pth"
    if name.endswith(marker):
        return name[: -len(marker)]
    marker = "_final_model.pth"
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


def discover_runs(runs_root, checkpoint_fragment):
    run_dirs = []
    for config_path in sorted(runs_root.glob("*/*/*/config_snapshot.yml")):
        run_dir = config_path.parent
        weights_dir = run_dir / "weights"
        if not weights_dir.exists():
            continue
        candidates = sorted(weights_dir.glob(f"*{checkpoint_fragment}*.pth"))
        if not candidates:
            continue
        model = parse_model_from_checkpoint(candidates[0])
        try:
            relative = run_dir.relative_to(runs_root)
            model_family = relative.parts[0]
            feature_family = relative.parts[1]
            run_name = relative.parts[2]
        except ValueError:
            model_family = model
            feature_family = ""
            run_name = run_dir.name
        apbs_variant = infer_apbs_variant(run_name)
        run_dirs.append(
            {
                "run_dir": run_dir,
                "run": run_name,
                "model": model,
                "model_family": model_family,
                "feature_family": feature_family,
                "apbs_variant": apbs_variant,
                "config": config_path,
                "checkpoint": candidates[0],
                "base_features": parse_base_features(run_dir),
            }
        )
    return run_dirs


def infer_apbs_variant(run_name):
    known = [
        "apbs_clip20_minmax",
        "apbs_full_signed",
        "apbs_posneg_clip20",
        "apbs_clip20_signed",
        "apbs_full_minmax",
        "apbs_no_cutoff_current",
        "apbs_clip10_minmax",
        "apbs_clip5_minmax",
    ]
    for variant in known:
        if run_name.endswith(variant):
            return variant
    match = re.search(r"(apbs_[a-z0-9_]+)$", run_name)
    return match.group(1) if match else ""


def clean_state_dict(state_dict, model_state_dict):
    has_module = any(key.startswith("module.") for key in state_dict)
    model_has_module = any(key.startswith("module.") for key in model_state_dict)
    if has_module and not model_has_module:
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    if model_has_module and not has_module:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return state_dict


def evaluate_run(run, device, num_workers, top_k_values, include_top_n_plus_2, reference_pocket_count):
    config = resolve_dataset_lists(load_config(str(run["config"])), str(run["config"]))
    set_reproducibility(int(config.get("seed", config.get("training", {}).get("seed", 42))))
    validation_config = config.get("validation", {})
    paper_config = validation_config.get("paper_metrics", {})

    dataset = ProteinLigandDatasetWithH5(
        h5_dir=config["h5_directory"],
        protein_names=config["datasets"].get("validation"),
        transform=build_transforms(config, training=False),
        config_path=str(run["config"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(validation_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
    )

    logger = logging.getLogger("work8a_topk")
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

    threshold = float(validation_config.get("threshold", 0.5))
    threshold_sweep = validation_config.get(
        "threshold_sweep",
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    thresholds = sorted({float(value) for value in [*threshold_sweep, threshold]})
    postprocess_mode = normalize_postprocess_mode(paper_config.get("postprocess", "raw"))
    dcc_cutoff = float(paper_config.get("dcc_cutoff_angstrom", 4.0))
    dca_cutoff = float(paper_config.get("dca_cutoff_angstrom", 4.0))
    min_component_voxels = int(paper_config.get("min_component_voxels", 5))
    min_component_volume = paper_config.get("min_component_volume_angstrom3")
    if min_component_volume is not None and postprocess_mode != "raw":
        min_component_volume = float(min_component_volume)
    else:
        min_component_volume = None

    run_context = {
        "epoch": "",
        "run": run["run"],
        "model": run["model_family"],
        "feature_family": run["feature_family"],
        "apbs_variant": run["apbs_variant"],
        "checkpoint": run["checkpoint"].name,
        "postprocess_mode": postprocess_mode,
    }
    component_rows = []
    topk_rows = []

    with torch.no_grad():
        for batch_idx, (protein, label) in enumerate(loader):
            protein = protein.to(device)
            output = model(protein).squeeze(1)
            probabilities = torch.sigmoid(output).detach().cpu().numpy()
            targets = label.detach().cpu().numpy()
            start_idx = batch_idx * int(validation_config.get("batch_size", 8))

            for sample_idx in range(probabilities.shape[0]):
                dataset_idx = start_idx + sample_idx
                protein_name = dataset.samples[dataset_idx][0]
                metadata = dataset.get_metadata(dataset_idx)
                ligand_mask = dataset.load_metric_mask(dataset_idx, "features", "ligand")
                sample_components, sample_topk = evaluate_topk_metrics_for_sample(
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
                    min_component_volume_angstrom3=min_component_volume,
                    postprocess_mode=postprocess_mode,
                    top_k_values=top_k_values,
                    reference_pocket_count=reference_pocket_count,
                    include_top_n_plus_2=include_top_n_plus_2,
                )
                component_rows.extend({**run_context, **row} for row in sample_components)
                topk_rows.extend({**run_context, **row} for row in sample_topk)

    return component_rows, topk_rows


def select_best_rows(summary_rows):
    grouped = {}
    for row in summary_rows:
        key = (row["run"], row["checkpoint"], row["top_k_label"])
        current = grouped.get(key)
        if current is None or best_key(row) > best_key(current):
            grouped[key] = row
    return sorted(grouped.values(), key=best_key, reverse=True)


def best_key(row):
    return (
        float(row["pocket_f1"]),
        float(row["dcc_success_rate_4a"]),
        float(row["dca_success_rate_4a"]),
        float(row["mean_dvo_of_best_dcc_success"]),
        float(row["mean_best_dvo_all"]),
        -float(row["mean_best_dcc_angstrom"]),
    )


def write_markdown_report(output_dir, best_rows, total_runs):
    report_path = output_dir / "work8a_topk_report.md"
    top3_rows = [row for row in best_rows if row["top_k_label"] == "top3"][:10]
    topn_rows = [row for row in best_rows if row["top_k_label"] == "top_n_plus_2"][:10]

    def line(row):
        return (
            f"| {row['model']} | {row['feature_family']} | {row['apbs_variant']} | "
            f"{float(row['threshold']):.2f} | {float(row['pocket_f1']):.4f} | "
            f"{float(row['dcc_success_rate_4a']):.4f} | {float(row['dca_success_rate_4a']):.4f} | "
            f"{float(row['mean_dvo_of_best_dcc_success']):.4f} | "
            f"{float(row['mean_best_dvo_all']):.4f} | "
            f"{float(row['best_dvo_dcc_success_rate_4a']):.4f} | "
            f"{float(row['best_dvo_dca_success_rate_4a']):.4f} | "
            f"{float(row['mean_dcc_of_best_dvo_angstrom']):.2f} | "
            f"{float(row['mean_dca_of_best_dvo_angstrom']):.2f} |"
        )

    lines = [
        "# Work8A Top-k Metric Re-evaluation",
        "",
        f"Evaluated runs: {total_runs}",
        "",
        "This report evaluates completed Work8 checkpoints without retraining. Top-k means the metric is allowed to choose the best matching pocket among the first k connected prediction components.",
        "",
        "Files:",
        "",
        "- `topk_summary_by_threshold.csv`: every run, threshold, and Top-k protocol.",
        "- `topk_best_by_run.csv`: best threshold per run and Top-k protocol.",
        "- `topk_per_protein.csv`: per-protein Top-k metrics.",
        "- `topk_component_metrics.csv`: component-level centers, DCC, DCA, and DVO.",
        "",
        "## Top-3 Best Rows",
        "",
        "| Model | Feature set | APBS variant | threshold | Pocket-F1 | DCC@4A | DCA@4A | DVO(success) | Best-DVO | Best-DVO DCC@4A | Best-DVO DCA@4A | Best-DVO mean DCC | Best-DVO mean DCA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(line(row) for row in top3_rows)
    lines.extend(
        [
            "",
            "## Top-(n+2) Best Rows",
            "",
            "For the current scPDB cache `n=1`, so Top-(n+2) is equivalent to Top-3 unless a future dataset provides multiple reference pockets per protein.",
            "",
            "| Model | Feature set | APBS variant | threshold | Pocket-F1 | DCC@4A | DCA@4A | DVO(success) | Best-DVO | Best-DVO DCC@4A | Best-DVO DCA@4A | Best-DVO mean DCC | Best-DVO mean DCA |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(line(row) for row in topn_rows)
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    top_k_values = tuple(sorted({int(item.strip()) for item in args.top_k.split(",") if item.strip()}))
    device = choose_device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    runs = discover_runs(runs_root, args.checkpoint_name)
    if args.limit_runs:
        runs = runs[: args.limit_runs]
    if not runs:
        raise SystemExit(f"No Work8 runs found under {runs_root}")

    logging.info("Device: %s", device)
    logging.info("Discovered %d runs", len(runs))
    all_component_rows = []
    all_topk_rows = []
    for idx, run in enumerate(runs, start=1):
        logging.info(
            "[%d/%d] %s | %s | %s",
            idx,
            len(runs),
            run["model_family"],
            run["feature_family"],
            run["run"],
        )
        component_rows, topk_rows = evaluate_run(
            run=run,
            device=device,
            num_workers=args.num_workers,
            top_k_values=top_k_values,
            include_top_n_plus_2=not args.no_top_n_plus_2,
            reference_pocket_count=args.reference_pocket_count,
        )
        all_component_rows.extend(component_rows)
        all_topk_rows.extend(topk_rows)

    summary_rows = summarize_topk_pocket_metrics(all_topk_rows)
    best_rows = select_best_rows(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "topk_component_metrics.csv", POCKET_TOPK_COMPONENT_FIELDNAMES, all_component_rows)
    write_rows(output_dir / "topk_per_protein.csv", POCKET_TOPK_PER_PROTEIN_FIELDNAMES, all_topk_rows)
    write_rows(output_dir / "topk_summary_by_threshold.csv", POCKET_TOPK_SUMMARY_FIELDNAMES, summary_rows)
    write_rows(output_dir / "topk_best_by_run.csv", POCKET_TOPK_SUMMARY_FIELDNAMES, best_rows)
    report_path = write_markdown_report(output_dir, best_rows, len(runs))
    logging.info("Wrote report: %s", report_path)
    logging.info("Wrote CSV outputs under: %s", output_dir)


if __name__ == "__main__":
    main()
