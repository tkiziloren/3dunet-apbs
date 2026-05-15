#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

H5_DIR="${H5_DIR:-/Users/tevfik/Sandbox/github/PHD/data/scPDB_cache_gridfix_v1/label_cavity6/box36_span70}"
SPLIT_SEED="${SPLIT_SEED:-43}"
SPLIT_FOLDS="${SPLIT_FOLDS:-5}"
WORK2_FOLDS="${WORK2_FOLDS:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
EPOCHS="${EPOCHS:-150}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
MODEL="${MODEL:-UNet3D4LStrided}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-$CONFIGURABLE_DIR/config/local/scpdb_kalasanty36/scpdb_kalasanty36_apbs_compact_chem_fold0.yml}"
DRY_RUN="${DRY_RUN:-0}"

WORK_NAME="${WORK_NAME:-work2_box36_top5_plus_apbs_newsplit_seed${SPLIT_SEED}_fold${WORK2_FOLDS//,/}_150epoch_thr${VALIDATION_THRESHOLD/./}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/$WORK_NAME}"
SPLIT_DIR="${SPLIT_DIR:-$H5_DIR/splits_cache_kfold${SPLIT_FOLDS}_seed${SPLIT_SEED}}"
CONFIG_DIR="${CONFIG_DIR:-$OUTPUT_ROOT/generated_configs}"

# Work 1 top five feature families plus APBS-only control.
RUN_FILTER="${RUN_FILTER:-apbs_shape_selected_chem,apbs_shape,apbs_shape_selected_chem_surface_hydro,shape_selected_chem,shape_only,apbs_only}"

cd "$CONFIGURABLE_DIR"

echo "Work 2 output root: $OUTPUT_ROOT"
echo "H5 dir: $H5_DIR"
echo "Split dir: $SPLIT_DIR"
echo "Split seed: $SPLIT_SEED"
echo "Work 2 folds: $WORK2_FOLDS"
echo "Feature sets: $RUN_FILTER"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo "Dry run: $DRY_RUN"
echo

if [[ ! -f "$SPLIT_DIR/fold0_train_cases.txt" ]]; then
  echo "Split files not found. Building new k-fold split..."
  "$PYTHON_BIN" scripts/build_cache_kfold_splits.py \
    --h5-dir "$H5_DIR" \
    --folds "$SPLIT_FOLDS" \
    --seed "$SPLIT_SEED" \
    --output-dir "$SPLIT_DIR"
else
  echo "Using existing split files."
fi

echo
echo "Starting Work 2 training sequence."
echo "Master log suggestion:"
echo "  $OUTPUT_ROOT/master.log"
echo

EPOCHS="$EPOCHS" \
EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
MODEL="$MODEL" \
BASE_FEATURES="$BASE_FEATURES" \
NUM_WORKERS="$NUM_WORKERS" \
PYTHON_BIN="$PYTHON_BIN" \
BASE_CONFIG="$BASE_CONFIG" \
SPLIT_DIR="$SPLIT_DIR" \
FOLDS="$WORK2_FOLDS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
CONFIG_DIR="$CONFIG_DIR" \
RUN_FILTER="$RUN_FILTER" \
SKIP_COMPLETED="${SKIP_COMPLETED:-1}" \
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}" \
DRY_RUN="$DRY_RUN" \
scripts/run_scpdb_cache_kfold_feature_ablation.sh
