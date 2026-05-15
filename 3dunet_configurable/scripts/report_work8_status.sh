#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040}"
MODELS="${MODELS:-ResNet3D4L,UNet3D4LA,UNetPlusPlus3D,CBAMUNet3D,ResNet3D4LGN}"
FEATURE_SETS="${FEATURE_SETS:-apbs_shape,apbs_shape_selected_chem}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip20_minmax,apbs_full_signed,apbs_posneg_clip20}"

export OUTPUT_ROOT MODELS FEATURE_SETS CUTOFF_VARIANTS

python3 - <<'PY'
import csv
import os
import re
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
models = [item.strip() for item in os.environ["MODELS"].split(",") if item.strip()]
feature_sets = [item.strip() for item in os.environ["FEATURE_SETS"].split(",") if item.strip()]
variants = [item.strip() for item in os.environ["CUTOFF_VARIANTS"].split(",") if item.strip()]
total = len(models) * len(feature_sets) * len(variants)

def config_name(feature_set, variant):
    return f"scpdb_apbs_cutoff_fold1_{feature_set}_{variant}"


def log_path(model, feature_set, variant):
    return root / model / feature_set / config_name(feature_set, variant) / "log" / "training.log"


def read_last_epoch(log_file):
    if not log_file.exists():
        return None
    last_epoch = None
    pattern = re.compile(r"Epoch \[(\d+)/(\d+)\], Iteration")
    with log_file.open(errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                last_epoch = int(match.group(1))
    return last_epoch


def is_completed(log_file):
    if not log_file.exists():
        return False
    last_epoch = read_last_epoch(log_file)
    if last_epoch is not None and last_epoch >= 250:
        return True
    text_tail = log_file.read_text(errors="replace")[-20000:]
    return "Epoch 250 validation summary" in text_tail


def read_best_metrics(log_file):
    metrics_path = log_file.parent / "validation_paper_metrics.csv"
    if not metrics_path.exists():
        return {}
    rows = []
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                score = float(row.get("selection_score", "nan"))
            except ValueError:
                score = float("nan")
            rows.append((score, row))
    if not rows:
        return {}
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[0][1]


completed = []
running_or_incomplete = []
not_started = []

for model in models:
    for feature_set in feature_sets:
        for variant in variants:
            log_file = log_path(model, feature_set, variant)
            name = config_name(feature_set, variant)
            if not log_file.exists():
                not_started.append((model, feature_set, variant, name, None, {}))
                continue
            last_epoch = read_last_epoch(log_file)
            metrics = read_best_metrics(log_file)
            if is_completed(log_file):
                completed.append((model, feature_set, variant, name, last_epoch, metrics))
            else:
                running_or_incomplete.append((model, feature_set, variant, name, last_epoch, metrics))

print(f"Output root: {root}")
print(f"Total planned trainings: {total}")
print(f"Completed trainings: {len(completed)}")
print(f"Currently running or incomplete: {len(running_or_incomplete)}")
print(f"Remaining not started: {len(not_started)}")

if running_or_incomplete:
    print()
    print("Running or incomplete trainings:")
    for model, feature_set, variant, name, last_epoch, _ in running_or_incomplete:
        epoch_text = f"{last_epoch}/250" if last_epoch is not None else "unknown"
        print(f"- {model} | {feature_set} | {variant} | epoch={epoch_text}")

if not_started:
    print()
    print("Not started trainings:")
    for model, feature_set, variant, _, _, _ in not_started:
        print(f"- {model} | {feature_set} | {variant}")

if completed:
    print()
    print("Completed trainings sorted by selection score:")
    rows = []
    for model, family, variant, name, last_epoch, row in completed:
        try:
            score = float(row.get("selection_score", "nan"))
        except ValueError:
            score = float("nan")
        rows.append((score, model, family, variant, name, last_epoch, row))
    for idx, (score, model, family, variant, name, last_epoch, row) in enumerate(sorted(rows, reverse=True), start=1):
        def num(key, default=0.0):
            try:
                return float(row.get(key, default))
            except ValueError:
                return default

        print(
            f"{idx:02d}. {model} | {family} | {variant} | "
            f"selection={score:.4f} | pocketF1={num('paper_f1'):.4f} | "
            f"DCC={num('dcc_success_rate_4a'):.4f} | "
            f"DCA={num('dca_success_rate_4a'):.4f} | "
            f"DVO={num('mean_dvo_dcc_success'):.4f} | "
            f"best_epoch={row.get('epoch', '')} | threshold={row.get('threshold', '')}"
        )
PY
