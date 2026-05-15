#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work8_top3_models_apbs_feature_sweep_fold1_250epoch_thr040}"
MODELS="${MODELS:-ResNet3D4L,UNet3D4LA,UNetPlusPlus3D}"
FEATURE_SETS="${FEATURE_SETS:-apbs_only,apbs_shape,apbs_selected_chem,apbs_shape_selected_chem}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip20}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-250}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_ROOT"

echo "Work8 top-model feature sweep"
echo "Output root: $OUTPUT_ROOT"
echo "Models: $MODELS"
echo "Feature sets: $FEATURE_SETS"
echo "APBS variant: $CUTOFF_VARIANTS"

IFS=',' read -r -a models <<< "$MODELS"
IFS=',' read -r -a feature_sets <<< "$FEATURE_SETS"

for model_name in "${models[@]}"; do
  model_name="$(echo "$model_name" | xargs)"
  [[ -z "$model_name" ]] && continue

  for feature_set in "${feature_sets[@]}"; do
    feature_set="$(echo "$feature_set" | xargs)"
    [[ -z "$feature_set" ]] && continue

    run_root="$OUTPUT_ROOT/$model_name/$feature_set"
    mkdir -p "$run_root"

    echo
    echo "============================================================"
    echo "Model: $model_name | Feature set: $feature_set"
    echo "============================================================"

    OUTPUT_ROOT="$run_root" \
    CONFIG_DIR="$run_root/generated_configs" \
    MODEL="$model_name" \
    FEATURE_SET="$feature_set" \
    CUTOFF_VARIANTS="$CUTOFF_VARIANTS" \
    FOLDS="$FOLDS" \
    EPOCHS="$EPOCHS" \
    EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
    VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
    BASE_FEATURES="$BASE_FEATURES" \
    NUM_WORKERS="$NUM_WORKERS" \
    PYTHON_BIN="$PYTHON_BIN" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    CLEAN_INCOMPLETE="$CLEAN_INCOMPLETE" \
    DRY_RUN="$DRY_RUN" \
    "$SCRIPT_DIR/run_apbs_cutoff_sweep.sh"
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run completed. Generated configs only; no training was started."
  exit 0
fi

echo
echo "============================================================"
echo "Combining Work8 summaries"
echo "============================================================"

export OUTPUT_ROOT MODELS FEATURE_SETS PYTHON_BIN
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
models = [item.strip() for item in os.environ["MODELS"].split(",") if item.strip()]
feature_sets = [item.strip() for item in os.environ["FEATURE_SETS"].split(",") if item.strip()]
rows = []
fieldnames = None

for model in models:
    for feature_set in feature_sets:
        summary = root / model / feature_set / "run_summary.csv"
        if not summary.exists():
            print(f"Missing summary: {summary}")
            continue
        with summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = {"model": model, "feature_family": feature_set, **row}
                rows.append(row)
                if fieldnames is None:
                    fieldnames = list(row.keys())

if not rows:
    raise SystemExit(f"No completed summaries found under {root}")

def as_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return -1.0

rows.sort(
    key=lambda row: (
        as_float(row, "paper_selection_score"),
        as_float(row, "paper_f1"),
        as_float(row, "voxel_f1"),
    ),
    reverse=True,
)

out = root / "run_summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary CSV: {out}")
for idx, row in enumerate(rows[:20], start=1):
    print(
        f"{idx},{row['model']},{row['feature_family']},"
        f"{row.get('paper_selection_score','')},{row.get('paper_f1','')},"
        f"{row.get('paper_dcc4','')},{row.get('paper_dvo_success','')},"
        f"{row.get('voxel_f1','')},{row.get('paper_threshold','')}"
    )
PY

echo "Done."
