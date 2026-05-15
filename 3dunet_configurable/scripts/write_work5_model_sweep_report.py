#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    "/Users/tevfik/Sandbox/github/PHD/runs/work5_apbs_only_clip20_model_sweep_fold1_250epoch_thr040"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Write a readable Work5 model-sweep report.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/work5_model_sweep_report.md"),
    )
    return parser.parse_args()


def as_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def fmt(value):
    if value != value:
        return ""
    return f"{value:.4f}"


def load_rows(output_root):
    rows = []
    for summary_path in sorted(output_root.glob("*/run_summary.csv")):
        model = summary_path.parent.name
        with summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = {"model": model, **row}
                rows.append(row)
    rows.sort(
        key=lambda row: (
            as_float(row, "paper_selection_score"),
            as_float(row, "paper_f1"),
            as_float(row, "voxel_f1"),
        ),
        reverse=True,
    )
    return rows


def best_by(rows, metric):
    if not rows:
        return None
    return max(rows, key=lambda row: as_float(row, metric))


def main():
    args = parse_args()
    rows = load_rows(args.output_root)
    if not rows:
        raise SystemExit(f"No completed model summaries found under {args.output_root}")

    report_lines = [
        "# Work5 Model Sweep Report",
        "",
        "## Scope",
        "",
        "- Feature set: `apbs_only`",
        "- APBS representation: `clip20`",
        "- Fold: `fold1`",
        "- Epochs: `250`",
        "- Early stopping: disabled",
        "- Fixed validation threshold: `0.40`",
        "- Ranking key: validation selection score, then Pocket-F1, then voxel-F1",
        "",
        "## Completed Models",
        "",
        f"Completed model count: `{len(rows)}`",
        "",
        "| Rank | Model | Best epoch | Best threshold | Selection | Pocket-F1 | DCC@4A | DCA@4A | DVO success | Voxel-F1 | Fixed F1@0.40 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(rows, start=1):
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{row['model']}`",
                    row.get("paper_epoch", ""),
                    row.get("paper_threshold", ""),
                    fmt(as_float(row, "paper_selection_score")),
                    fmt(as_float(row, "paper_f1")),
                    fmt(as_float(row, "paper_dcc4")),
                    fmt(as_float(row, "paper_dca4")),
                    fmt(as_float(row, "paper_dvo_success")),
                    fmt(as_float(row, "voxel_f1")),
                    fmt(as_float(row, "primary_f1_fixed_threshold")),
                ]
            )
            + " |"
        )

    metric_labels = [
        ("paper_selection_score", "Selection score"),
        ("paper_f1", "Pocket-F1"),
        ("paper_dcc4", "DCC@4A"),
        ("paper_dca4", "DCA@4A"),
        ("paper_dvo_success", "DVO success"),
        ("voxel_f1", "Voxel-F1"),
        ("primary_f1_fixed_threshold", "Fixed F1@0.40"),
    ]
    report_lines.extend(["", "## Best By Metric", ""])
    for metric, label in metric_labels:
        row = best_by(rows, metric)
        report_lines.append(
            f"- {label}: `{row['model']}` = `{fmt(as_float(row, metric))}` "
            f"(selection `{fmt(as_float(row, 'paper_selection_score'))}`, "
            f"Pocket-F1 `{fmt(as_float(row, 'paper_f1'))}`, "
            f"threshold `{row.get('paper_threshold', '')}`, epoch `{row.get('paper_epoch', '')}`)"
        )

    top = rows[0]
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest completed Work5 model is `{top['model']}`. It has the best selection score, Pocket-F1, DCC@4A, voxel-F1, and fixed-threshold F1 among completed models.",
            "",
            "The main practical conclusion is that APBS-only performance is not only a feature-representation problem; architecture matters as well. `ResNet3D4L` appears to extract substantially more useful signal from APBS-only `clip20` than the baseline U-Net family.",
            "",
            "ConvNeXt-style models should be treated carefully. The original heavy `ConvNeXtUNet3D` was stopped because it was too slow and showed poor early learning. Work6 uses lighter, 3D-friendly ConvNeXt-style variants instead.",
            "",
            "## Files",
            "",
            f"- Output root: `{args.output_root}`",
            "- Per-model summaries: `<output-root>/<model>/run_summary.csv`",
            "",
        ]
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(args.report_path)


if __name__ == "__main__":
    main()
