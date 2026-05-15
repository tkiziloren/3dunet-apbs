#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work4_apbs_clip20_combined_features_fold0_250epoch_thr040}"
FEATURE_SETS="${FEATURE_SETS:-apbs_shape,apbs_shape_selected_chem}"
FOLDS="${FOLDS:-0}"
EPOCHS="${EPOCHS:-250}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip20}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_ROOT"

echo "Work4 output root: $OUTPUT_ROOT"
echo "Feature sets: $FEATURE_SETS"
echo "Folds: $FOLDS"
echo "Epochs: $EPOCHS"
echo "Cutoff variants: $CUTOFF_VARIANTS"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Dry run: $DRY_RUN"

IFS=',' read -r -a feature_sets <<< "$FEATURE_SETS"

for feature_set in "${feature_sets[@]}"; do
  feature_set="$(echo "$feature_set" | xargs)"
  [[ -z "$feature_set" ]] && continue

  echo
  echo "============================================================"
  echo "Work4 feature set: $feature_set"
  echo "============================================================"

  OUTPUT_ROOT="$OUTPUT_ROOT" \
  CONFIG_DIR="$OUTPUT_ROOT/generated_configs/$feature_set" \
  FEATURE_SET="$feature_set" \
  FOLDS="$FOLDS" \
  EPOCHS="$EPOCHS" \
  EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
  VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
  CUTOFF_VARIANTS="$CUTOFF_VARIANTS" \
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
echo "Final Work4 summary"
echo "============================================================"
"${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}" "$CONFIGURABLE_DIR/scripts/summarize_gridfix_runs.py" \
  --output-root "$OUTPUT_ROOT" \
  --write-csv "$OUTPUT_ROOT/run_summary.csv" | tee "$OUTPUT_ROOT/summary_stdout.txt"

echo
echo "Done."
echo "Summary CSV: $OUTPUT_ROOT/run_summary.csv"
echo "Master output can be redirected to: $OUTPUT_ROOT/master.log"

