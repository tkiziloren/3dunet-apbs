import argparse
import csv
import os
import re
from pathlib import Path


FLOAT_COLUMNS = {
    "primary_voxel_f1",
    "primary_voxel_precision",
    "primary_voxel_recall",
    "best_voxel_threshold",
    "best_voxel_f1",
    "best_voxel_precision",
    "best_voxel_recall",
    "best_paper_threshold",
    "best_paper_f1",
    "best_paper_precision",
    "best_paper_recall",
    "best_paper_dcc_success_rate_4a",
    "best_paper_dca_success_rate_4a",
    "best_paper_dvo_all",
    "best_paper_dvo_dcc_success",
    "best_paper_pli_all",
    "best_paper_pli_dcc_success",
    "best_paper_mean_dcc_angstrom",
    "best_paper_mean_predicted_positive_voxels",
    "best_paper_selection_score",
    "paper_f1_fixed_threshold_040",
    "paper_f1_fixed_threshold_050",
}

INT_COLUMNS = {
    "epoch",
    "primary_voxel_tp",
    "primary_voxel_fp",
    "primary_voxel_tn",
    "primary_voxel_fn",
    "best_voxel_tp",
    "best_voxel_fp",
    "best_voxel_tn",
    "best_voxel_fn",
    "best_paper_dcc_success_count",
    "best_paper_no_prediction_count",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize local gridfix training run CSV outputs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write-csv", default=None)
    return parser.parse_args()


def as_float(row, key, default=-1.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def load_best_threshold_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fixed_paper_f1_for_epoch(run_dir: Path, epoch, threshold):
    paper_path = run_dir / "log" / "validation_paper_metrics.csv"
    if not paper_path.exists():
        return ""
    epoch = str(epoch)
    threshold = float(threshold)
    with paper_path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("epoch") != epoch:
                continue
            try:
                row_threshold = float(row.get("threshold", "nan"))
            except ValueError:
                continue
            if abs(row_threshold - threshold) < 1e-9:
                return row.get("paper_f1", "")
    return ""


def parse_config_from_log(log_path: Path):
    info = {}
    if not log_path.exists():
        return info

    patterns = {
        "configuration_name": re.compile(r"Configuration name:\s*(.+)$"),
        "feature_set": re.compile(r"Feature set:\s*(.+)$"),
        "label": re.compile(r"Label:\s*(.+)$"),
        "features": re.compile(r"Features:\s*(.+)$"),
        "loss": re.compile(r"Loss function:\s*(.+)$"),
        "fixed_pos_weight": re.compile(r"Using fixed pos_weight from config:\s*(.+)$"),
        "dynamic_pos_weight": re.compile(r"Using dynamically calculated pos_weight:\s*(.+)$"),
    }
    with log_path.open(errors="replace") as handle:
        for line in handle:
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    info[key] = match.group(1).strip()
    return info


def best_rows_for_run(run_dir: Path):
    best_path = run_dir / "log" / "validation_best_thresholds.csv"
    if not best_path.exists():
        return None

    rows = load_best_threshold_rows(best_path)
    if not rows:
        return None

    best_by_paper = max(rows, key=lambda row: as_float(row, "best_paper_selection_score"))
    best_by_voxel = max(rows, key=lambda row: as_float(row, "best_voxel_f1"))
    best_by_primary = max(rows, key=lambda row: as_float(row, "primary_voxel_f1"))
    log_info = parse_config_from_log(run_dir / "log" / "training.log")
    fold_match = re.search(r"fold(\d+)", run_dir.name)
    paper_f1_at_040 = best_by_paper.get("paper_f1_fixed_threshold_040", "")
    paper_f1_at_050 = best_by_paper.get("paper_f1_fixed_threshold_050", "")
    if paper_f1_at_040 == "":
        paper_f1_at_040 = fixed_paper_f1_for_epoch(run_dir, best_by_paper["epoch"], 0.4)
    if paper_f1_at_050 == "":
        paper_f1_at_050 = fixed_paper_f1_for_epoch(run_dir, best_by_paper["epoch"], 0.5)

    summary = {
        "run": run_dir.name,
        "configuration_name": log_info.get("configuration_name", ""),
        "fold": fold_match.group(1) if fold_match else "",
        "feature_set": log_info.get("feature_set", ""),
        "label": log_info.get("label", ""),
        "features": log_info.get("features", ""),
        "loss": log_info.get("loss", ""),
        "pos_weight": log_info.get("fixed_pos_weight") or log_info.get("dynamic_pos_weight", ""),
        "paper_epoch": best_by_paper["epoch"],
        "paper_threshold": best_by_paper.get("best_paper_threshold", ""),
        "paper_selection_score": best_by_paper.get("best_paper_selection_score", ""),
        "paper_f1": best_by_paper.get("best_paper_f1", ""),
        "paper_f1_fixed_threshold_040": paper_f1_at_040,
        "paper_f1_fixed_threshold_050": paper_f1_at_050,
        "paper_dcc4": best_by_paper.get("best_paper_dcc_success_rate_4a", ""),
        "paper_dca4": best_by_paper.get("best_paper_dca_success_rate_4a", ""),
        "paper_dvo_all": best_by_paper.get("best_paper_dvo_all", ""),
        "paper_dvo_success": best_by_paper.get("best_paper_dvo_dcc_success", ""),
        "paper_pli_all": best_by_paper.get("best_paper_pli_all", ""),
        "paper_pli_success": best_by_paper.get("best_paper_pli_dcc_success", ""),
        "paper_mean_dcc": best_by_paper.get("best_paper_mean_dcc_angstrom", ""),
        "paper_mean_pred_voxels": best_by_paper.get("best_paper_mean_predicted_positive_voxels", ""),
        "voxel_epoch": best_by_voxel["epoch"],
        "voxel_threshold": best_by_voxel.get("best_voxel_threshold", ""),
        "voxel_f1": best_by_voxel.get("best_voxel_f1", ""),
        "voxel_precision": best_by_voxel.get("best_voxel_precision", ""),
        "voxel_recall": best_by_voxel.get("best_voxel_recall", ""),
        "primary_epoch": best_by_primary["epoch"],
        "primary_threshold": best_by_primary.get("primary_threshold", ""),
        "primary_f1_fixed_threshold": best_by_primary.get("primary_voxel_f1", ""),
        "primary_precision_fixed_threshold": best_by_primary.get("primary_voxel_precision", ""),
        "primary_recall_fixed_threshold": best_by_primary.get("primary_voxel_recall", ""),
    }
    return summary


def sort_key(row):
    return (
        as_float(row, "paper_selection_score"),
        as_float(row, "paper_f1"),
        as_float(row, "voxel_f1"),
    )


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    summaries = []

    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        summary = best_rows_for_run(child)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=sort_key, reverse=True)
    if not summaries:
        raise SystemExit(f"No completed runs found under {output_root}")

    fieldnames = list(summaries[0].keys())
    csv_path = Path(args.write_csv) if args.write_csv else output_root / "run_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Summary CSV: {csv_path}")
    print(
        "rank,run,fold,feature_set,label,pos_weight,paper_score,paper_f1,paper_threshold,"
        "paper_f1_at_040,paper_f1_at_050,dcc4,dca4,dvo_success,pli_success,"
        "voxel_f1,voxel_threshold,primary_threshold,primary_f1_fixed"
    )
    for idx, row in enumerate(summaries, start=1):
        print(
            f"{idx},{row['run']},{row['fold']},{row['feature_set']},{row['label']},{row['pos_weight']},"
            f"{row['paper_selection_score']},{row['paper_f1']},{row['paper_threshold']},"
            f"{row['paper_f1_fixed_threshold_040']},{row['paper_f1_fixed_threshold_050']},"
            f"{row['paper_dcc4']},{row['paper_dca4']},{row['paper_dvo_success']},"
            f"{row['paper_pli_success']},{row['voxel_f1']},{row['voxel_threshold']},"
            f"{row['primary_threshold']},{row['primary_f1_fixed_threshold']}"
        )


if __name__ == "__main__":
    main()
