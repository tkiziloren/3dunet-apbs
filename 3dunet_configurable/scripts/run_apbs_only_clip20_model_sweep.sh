#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS="${MODELS:-UNet3D4LStrided,UNet3D4LAStrided,UNet3D4L,UNet3D4LA,UNet3D5L,ResidualUNet3D,SEResUNet3D,CBAMUNet3D,LightweightUNet3D,UNetPlusPlus3D,ResNet3D4L,ResNet3D5L,ConvNeXtUNet3D,ConvNeXt3D}"
FEATURE_SET="${FEATURE_SET:-apbs_only}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip20}"
FOLDS="${FOLDS:-0}"
EPOCHS="${EPOCHS:-250}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/apbs_only_clip20_model_sweep_fold${FOLDS//,/}_${EPOCHS}epoch_thr${VALIDATION_THRESHOLD/./}}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"

mkdir -p "$OUTPUT_ROOT"

echo "APBS-only clip20 model sweep output root: $OUTPUT_ROOT"
echo "Models: $MODELS"
echo "Feature set: $FEATURE_SET"
echo "Cutoff variants: $CUTOFF_VARIANTS"
echo "Folds: $FOLDS"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Base features: $BASE_FEATURES"
echo "Dry run: $DRY_RUN"

IFS=',' read -r -a models <<< "$MODELS"

for model_name in "${models[@]}"; do
  model_name="$(echo "$model_name" | xargs)"
  [[ -z "$model_name" ]] && continue

  model_output_root="$OUTPUT_ROOT/$model_name"

  echo
  echo "============================================================"
  echo "Model: $model_name"
  echo "============================================================"

  OUTPUT_ROOT="$model_output_root" \
  CONFIG_DIR="$model_output_root/generated_configs" \
  FEATURE_SET="$FEATURE_SET" \
  CUTOFF_VARIANTS="$CUTOFF_VARIANTS" \
  FOLDS="$FOLDS" \
  EPOCHS="$EPOCHS" \
  EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
  VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
  MODEL="$model_name" \
  BASE_FEATURES="$BASE_FEATURES" \
  NUM_WORKERS="$NUM_WORKERS" \
  PYTHON_BIN="$PYTHON_BIN" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  CLEAN_INCOMPLETE="$CLEAN_INCOMPLETE" \
  DRY_RUN="$DRY_RUN" \
  "$SCRIPT_DIR/run_apbs_cutoff_sweep.sh"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run completed. Generated configs only; no training was started."
  exit 0
fi

echo
echo "============================================================"
echo "Final APBS-only clip20 model sweep summary"
echo "============================================================"

export OUTPUT_ROOT MODELS PYTHON_BIN
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
models = [item.strip() for item in os.environ["MODELS"].split(",") if item.strip()]
rows = []
fieldnames = None

for model_name in models:
    summary_path = output_root / model_name / "run_summary.csv"
    if not summary_path.exists():
        print(f"Missing summary for {model_name}: {summary_path}")
        continue
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {"model": model_name, **row}
            rows.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())

if not rows:
    raise SystemExit(f"No completed model summaries found under {output_root}")

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

summary_path = output_root / "run_summary.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary CSV: {summary_path}")
print("rank,model,run,paper_score,paper_f1,threshold,dcc4,dca4,dvo_success,voxel_f1,fixed_f1")
for idx, row in enumerate(rows, start=1):
    print(
        f"{idx},{row['model']},{row['run']},"
        f"{row.get('paper_selection_score','')},{row.get('paper_f1','')},{row.get('paper_threshold','')},"
        f"{row.get('paper_dcc4','')},{row.get('paper_dca4','')},{row.get('paper_dvo_success','')},"
        f"{row.get('voxel_f1','')},{row.get('primary_f1_fixed_threshold','')}"
    )
PY

echo
echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
echo "Master output can be redirected to: $OUTPUT_ROOT/master.log"
