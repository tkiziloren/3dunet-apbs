#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/nfs/production/arl/chembl/tevfik/DEEP_APBS_DATASETS}"
ROOT="${ROOT:-$DATA_ROOT/runs/work22_scpdb_box36_span70_puresnet5020_kfold4_unetplusplus_selectedchem_plus_one_250epoch_thr040}"
JOB_ID="${JOB_ID:-28877841}"
MAX_EPOCH="${MAX_EPOCH:-250}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

SQUEUE=squeue
if ! command -v squeue >/dev/null 2>&1 && [[ -x /ebi/slurm/codon/bin/squeue ]]; then
  SQUEUE=/ebi/slurm/codon/bin/squeue
fi

echo
echo "================================================================================"
echo "SLURM QUEUE"
echo "================================================================================"
"$SQUEUE" -j "$JOB_ID" -o "%.18i %.9P %.22j %.8u %.2t %.10M %.6D %R" || true

python - "$ROOT" "$MAX_EPOCH" > "$tmp/work22.tsv" <<'PY'
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import re
import statistics
import sys

root = Path(sys.argv[1])
max_epoch = int(sys.argv[2])
config_list = root / "generated_configs" / "config_list.txt"

BEST_RE = re.compile(
    r"selection score ([0-9.]+) \| pocket F1 ([0-9.]+) \| threshold ([0-9.]+) \| epoch ([0-9]+)"
)
VAL_RE = re.compile(r"Epoch ([0-9]+) validation summary")
ITER_RE = re.compile(r"Epoch \[([0-9]+)/([0-9]+)\], Iteration \[([0-9]+)/([0-9]+)\], Loss: ([0-9.]+)")
STOP_RE = re.compile(r"Early stopping|Training stopped early|Stopping training|Training completed|Training complete")
TIME_RE = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})")


def fnum(value):
    try:
        if value in (None, "", "nan", "NaN", "inf", "Infinity"):
            return None
        return float(value)
    except Exception:
        return None


def pct(value):
    value = fnum(value)
    return None if value is None else value * 100.0


def fmt(value, digits=1):
    return "" if value is None else f"{float(value):.{digits}f}"


def fmt_sel(value):
    return "" if value is None else f"{float(value):.4f}"


def parse_dt(line: str):
    m = TIME_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    try:
        with path.open(newline="", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def close_float(a, b):
    aa = fnum(a)
    bb = fnum(b)
    return aa is not None and bb is not None and abs(aa - bb) < 1e-9


def parse_fold(run: str):
    m = re.search(r"_fold([0-9]+)_", run)
    return m.group(1) if m else ""


def parse_added_feature(run: str):
    m = re.search(r"_surface_plus_(.+?)_UNetPlusPlus3D_", run)
    return m.group(1) if m else run


def expected_runs():
    rows = []
    if not config_list.exists():
        return rows
    for idx, line in enumerate(config_list.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        run = Path(line.strip()).stem
        rows.append(
            {
                "task": str(idx),
                "fold": parse_fold(run),
                "feature": parse_added_feature(run),
                "run": run,
            }
        )
    return rows


def best_metric_row(run_dir: Path):
    rows = read_csv_rows(run_dir / "log" / "validation_paper_metrics.csv")
    best = None
    best_score = None
    for row in rows:
        score = fnum(row.get("selection_score"))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best = row
            best_score = score
    return best


def topk_rows_at_best(run_dir: Path, best_row):
    source_rows = []
    for path in [
        run_dir / "log" / "validation_paper_metrics_topk_kalasanty_puresnet.csv",
        run_dir / "log" / "validation_paper_metrics_topk.csv",
    ]:
        source_rows = read_csv_rows(path)
        if source_rows:
            break

    picked = {"top1": None, "top3": None}
    if not source_rows or not best_row:
        return picked

    best_epoch = best_row.get("epoch")
    best_thr = best_row.get("threshold")
    candidates = [
        row
        for row in source_rows
        if str(row.get("epoch")) == str(best_epoch) and close_float(row.get("threshold"), best_thr)
    ]
    if not candidates:
        candidates = [row for row in source_rows if str(row.get("epoch")) == str(best_epoch)]

    for row in candidates:
        label = str(row.get("top_k_label") or "").lower()
        top_k = str(row.get("top_k") or "")
        if label == "top1" or top_k == "1":
            picked["top1"] = row
        if label == "top3" or top_k == "3":
            picked["top3"] = row
    return picked


def field(row, name):
    if row is None:
        return None
    return row.get(name)


def metric_pack(run_dir: Path, log_best):
    best_row = best_metric_row(run_dir)
    best_sel = None
    best_epoch = None
    best_thr = None

    if best_row:
        best_sel = fnum(best_row.get("selection_score"))
        best_epoch = best_row.get("epoch")
        best_thr = fnum(best_row.get("threshold"))
    if log_best:
        best_sel = log_best[0]
        best_thr = log_best[2]
        best_epoch = log_best[3]

    topk = topk_rows_at_best(run_dir, best_row)
    top1 = topk["top1"]
    top3 = topk["top3"]
    if top1 is None and best_row is not None:
        top1 = {
            "pocket_f1": best_row.get("paper_f1"),
            "dcc_success_rate_4a": best_row.get("dcc_success_rate_4a"),
            "dca_success_rate_4a": best_row.get("dca_success_rate_4a"),
            "mean_dvo_of_best_dcc_success": best_row.get("mean_dvo_dcc_success"),
            "mean_pli_of_best_dcc_success": best_row.get("mean_pli_dcc_success"),
            "no_prediction_count": best_row.get("no_prediction_count"),
        }

    no_pred = fnum(field(top3, "no_prediction_count"))
    if no_pred is None:
        no_pred = fnum(field(top1, "no_prediction_count"))

    return {
        "best_sel": best_sel,
        "thr": best_thr,
        "best_ep": best_epoch,
        "t1_f1": pct(field(top1, "pocket_f1")),
        "t3_f1": pct(field(top3, "pocket_f1")),
        "t1_dcc": pct(field(top1, "dcc_success_rate_4a")),
        "t3_dcc": pct(field(top3, "dcc_success_rate_4a")),
        "t1_dca": pct(field(top1, "dca_success_rate_4a")),
        "t3_dca": pct(field(top3, "dca_success_rate_4a")),
        "t1_dvo": pct(field(top1, "mean_dvo_of_best_dcc_success")),
        "t3_dvo": pct(field(top3, "mean_dvo_of_best_dcc_success")),
        "t1_pli": pct(field(top1, "mean_pli_of_best_dcc_success")),
        "t3_pli": pct(field(top3, "mean_pli_of_best_dcc_success")),
        "no_pred": no_pred,
    }


by_run = {}
for log in root.glob("*/log/training.log"):
    run_dir = log.parents[1]
    run = run_dir.name
    text = log.read_text(errors="replace")
    lines = text.splitlines()

    log_best = None
    for line in lines:
        if "Checkpoint saved: new best validation selection score" in line:
            m = BEST_RE.search(line)
            if m:
                log_best = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)))

    metrics = metric_pack(run_dir, log_best)

    last_val = ""
    for line in lines:
        m = VAL_RE.search(line)
        if m:
            last_val = int(m.group(1))

    current = ""
    current_epoch = None
    for line in reversed(lines):
        m = ITER_RE.search(line)
        if m:
            current_epoch = int(m.group(1))
            current = "E{}/{} I{}/{} L{:.4f}".format(
                m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
            )
            break

    if "Traceback (most recent call last)" in text or "ValueError:" in text or "RuntimeError:" in text:
        status = "fail"
    elif f"Epoch {max_epoch} validation summary" in text or STOP_RE.search(text):
        status = "done"
    elif current:
        status = "run"
    else:
        status = "wait"

    first_dt = None
    last_dt = None
    for line in lines:
        dt = parse_dt(line)
        if dt and first_dt is None:
            first_dt = dt
        if dt:
            last_dt = dt
    ep_per_hour = ""
    if first_dt and last_dt and current_epoch:
        elapsed_min = max((last_dt - first_dt).total_seconds() / 60.0, 0.01)
        ep_per_hour = f"{current_epoch / elapsed_min * 60.0:.1f}"

    by_run[run] = {
        "task": "",
        "fold": parse_fold(run),
        "feature": parse_added_feature(run),
        "run": run,
        "status": status,
        "last_val": str(last_val),
        "current": current,
        "ep_per_hour": ep_per_hour,
        **metrics,
    }


rows = []
for item in expected_runs():
    row = dict(by_run.get(item["run"], {}))
    if not row:
        row = {
            "status": "wait",
            "last_val": "",
            "current": "",
            "ep_per_hour": "",
            "best_sel": None,
            "thr": None,
            "best_ep": "",
            "t1_f1": None,
            "t3_f1": None,
            "t1_dcc": None,
            "t3_dcc": None,
            "t1_dca": None,
            "t3_dca": None,
            "t1_dvo": None,
            "t3_dvo": None,
            "t1_pli": None,
            "t3_pli": None,
            "no_pred": None,
        }
    row["task"] = item["task"]
    row["fold"] = row.get("fold") or item["fold"]
    row["feature"] = row.get("feature") or item["feature"]
    row["run"] = item["run"]
    rows.append(row)


def sort_key(row):
    sel = row.get("best_sel")
    sel = -1.0 if sel is None else float(sel)
    try:
        task = int(row.get("task") or 9999)
    except Exception:
        task = 9999
    return (-sel, task)


def mean(rows, key):
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.mean(vals) if vals else None


summary_cols = [
    "items", "best", "done", "run", "fail", "wait", "sel", "T1F1%", "T3F1%",
    "T1DCC%", "T3DCC%", "T1DCA%", "T3DCA%", "T1DVO%", "T3DVO%", "T1PLI%", "T3PLI%",
    "no_pred", "ep/hr",
]
status_counts = defaultdict(int)
for row in rows:
    status_counts[row["status"]] += 1

print("__SUMMARY__")
print("\t".join(summary_cols))
print(
    "\t".join(
        [
            str(len(rows)),
            str(sum(1 for row in rows if row.get("best_sel") is not None)),
            str(status_counts["done"]),
            str(status_counts["run"]),
            str(status_counts["fail"]),
            str(status_counts["wait"]),
            fmt_sel(mean(rows, "best_sel")),
            fmt(mean(rows, "t1_f1")),
            fmt(mean(rows, "t3_f1")),
            fmt(mean(rows, "t1_dcc")),
            fmt(mean(rows, "t3_dcc")),
            fmt(mean(rows, "t1_dca")),
            fmt(mean(rows, "t3_dca")),
            fmt(mean(rows, "t1_dvo")),
            fmt(mean(rows, "t3_dvo")),
            fmt(mean(rows, "t1_pli")),
            fmt(mean(rows, "t3_pli")),
            fmt(mean(rows, "no_pred")),
            fmt(mean(rows, "ep_per_hour")),
        ]
    )
)


def grouped_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)

    out = []
    for name, group_rows in grouped.items():
        counts = defaultdict(int)
        for row in group_rows:
            counts[row.get("status", "")] += 1
        out.append(
            {
                "name": name,
                "n": len(group_rows),
                "best": sum(1 for row in group_rows if row.get("best_sel") is not None),
                "done": counts["done"],
                "run": counts["run"],
                "fail": counts["fail"],
                "wait": counts["wait"],
                "best_sel": mean(group_rows, "best_sel"),
                "t1_f1": mean(group_rows, "t1_f1"),
                "t3_f1": mean(group_rows, "t3_f1"),
                "t1_dcc": mean(group_rows, "t1_dcc"),
                "t3_dcc": mean(group_rows, "t3_dcc"),
                "t1_dca": mean(group_rows, "t1_dca"),
                "t3_dca": mean(group_rows, "t3_dca"),
                "t1_dvo": mean(group_rows, "t1_dvo"),
                "t3_dvo": mean(group_rows, "t3_dvo"),
                "t1_pli": mean(group_rows, "t1_pli"),
                "t3_pli": mean(group_rows, "t3_pli"),
                "no_pred": mean(group_rows, "no_pred"),
                "ep_per_hour": mean(group_rows, "ep_per_hour"),
            }
        )
    return out


group_cols = [
    "group", "items", "best", "done", "run", "fail", "wait", "sel", "T1F1%", "T3F1%",
    "T1DCC%", "T3DCC%", "T1DCA%", "T3DCA%", "T1DVO%", "T3DVO%", "T1PLI%", "T3PLI%",
    "no_pred", "ep/hr",
]


def print_group_table(marker, group_rows, sort_by_score=True):
    print(marker)
    print("\t".join(group_cols))
    if sort_by_score:
        ordered = sorted(
            group_rows,
            key=lambda row: (float(row["t3_f1"]) if row["t3_f1"] is not None else -1.0, str(row["name"])),
            reverse=True,
        )
    else:
        ordered = sorted(group_rows, key=lambda row: str(row["name"]))
    for row in ordered:
        print(
            "\t".join(
                [
                    str(row["name"]),
                    str(row["n"]),
                    str(row["best"]),
                    str(row["done"]),
                    str(row["run"]),
                    str(row["fail"]),
                    str(row["wait"]),
                    fmt_sel(row["best_sel"]),
                    fmt(row["t1_f1"]),
                    fmt(row["t3_f1"]),
                    fmt(row["t1_dcc"]),
                    fmt(row["t3_dcc"]),
                    fmt(row["t1_dca"]),
                    fmt(row["t3_dca"]),
                    fmt(row["t1_dvo"]),
                    fmt(row["t3_dvo"]),
                    fmt(row["t1_pli"]),
                    fmt(row["t3_pli"]),
                    fmt(row["no_pred"]),
                    fmt(row["ep_per_hour"]),
                ]
            )
        )


print_group_table("__FOLD_MEANS__", grouped_rows(rows, "fold"), sort_by_score=False)
print_group_table("__FEATURE_MEANS__", grouped_rows(rows, "feature"), sort_by_score=True)

print("__ROWS__")
row_cols = [
    "rank", "task", "fold", "added_feature", "sel", "T1F1%", "T3F1%", "T1DCC%", "T3DCC%",
    "T1DCA%", "T3DCA%", "T1DVO%", "T3DVO%", "T1PLI%", "T3PLI%", "no_pred",
    "thr", "best_ep", "last_val", "status", "current", "ep/hr",
]
print("\t".join(row_cols))
for rank, row in enumerate(sorted(rows, key=sort_key), 1):
    print(
        "\t".join(
            [
                str(rank),
                str(row.get("task", "")),
                str(row.get("fold", "")),
                str(row.get("feature", "")),
                fmt_sel(row.get("best_sel")),
                fmt(row.get("t1_f1")),
                fmt(row.get("t3_f1")),
                fmt(row.get("t1_dcc")),
                fmt(row.get("t3_dcc")),
                fmt(row.get("t1_dca")),
                fmt(row.get("t3_dca")),
                fmt(row.get("t1_dvo")),
                fmt(row.get("t3_dvo")),
                fmt(row.get("t1_pli")),
                fmt(row.get("t3_pli")),
                "" if row.get("no_pred") is None else str(int(row["no_pred"])),
                "" if row.get("thr") is None else f"{row['thr']:.2f}",
                "" if row.get("best_ep") is None else str(row.get("best_ep")),
                row.get("last_val", ""),
                row.get("status", ""),
                row.get("current", ""),
                row.get("ep_per_hour", ""),
            ]
        )
    )
PY

echo
echo "================================================================================"
echo "work22_puresnet5020_unetplusplus_selectedchem_plus_one_250epoch"
echo "Root: $ROOT"
echo "Job: $JOB_ID"
echo "Note: F1/DCC/DCA/DVO/PLI columns are percentages. T1=Top-1, T3=Top-3."
echo "================================================================================"
echo
echo "SUMMARY"
awk 'BEGIN{p=0} /^__SUMMARY__/{p=1; next} /^__FOLD_MEANS__/{p=0} p{print}' "$tmp/work22.tsv" | column -t -s $'\t'
echo
echo "FOLD MEANS"
awk 'BEGIN{p=0} /^__FOLD_MEANS__/{p=1; next} /^__FEATURE_MEANS__/{p=0} p{print}' "$tmp/work22.tsv" | column -t -s $'\t'
echo
echo "FEATURE MEANS"
awk 'BEGIN{p=0} /^__FEATURE_MEANS__/{p=1; next} /^__ROWS__/{p=0} p{print}' "$tmp/work22.tsv" | column -t -s $'\t'
echo
echo "RUNS"
awk 'BEGIN{p=0} /^__ROWS__/{p=1; next} p{print}' "$tmp/work22.tsv" | column -t -s $'\t'

echo
echo "================================================================================"
echo "QUICK LOG COMMANDS"
echo "================================================================================"
echo "tail -f \"$ROOT/slurm/scp36-scplus_${JOB_ID}_1.out\""
echo "tail -f \"$ROOT/slurm/scp36-scplus_${JOB_ID}_1.err\""
