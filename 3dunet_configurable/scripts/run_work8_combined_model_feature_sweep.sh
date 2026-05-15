#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040}"

# Work8 is a single combined-feature experiment.
# It keeps the matrix focused enough to interpret and broad enough to test model-feature interactions.
MODELS="${MODELS:-ResNet3D4L,UNet3D4LA,UNetPlusPlus3D,CBAMUNet3D,ResNet3D4LGN}"
FEATURE_SETS="${FEATURE_SETS:-apbs_shape,apbs_shape_selected_chem}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip20_minmax,apbs_full_signed,apbs_posneg_clip20}"

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

echo "Work8 combined model-feature-representation sweep"
echo "Output root: $OUTPUT_ROOT"
echo "Models: $MODELS"
echo "Feature sets: $FEATURE_SETS"
echo "APBS variants: $CUTOFF_VARIANTS"
echo "Folds: $FOLDS"
echo "Epochs: $EPOCHS"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Fixed validation threshold: $VALIDATION_THRESHOLD"
echo

MODELS="$MODELS" \
FEATURE_SETS="$FEATURE_SETS" \
CUTOFF_VARIANTS="$CUTOFF_VARIANTS" \
FOLDS="$FOLDS" \
EPOCHS="$EPOCHS" \
EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
BASE_FEATURES="$BASE_FEATURES" \
NUM_WORKERS="$NUM_WORKERS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SKIP_COMPLETED="$SKIP_COMPLETED" \
CLEAN_INCOMPLETE="$CLEAN_INCOMPLETE" \
DRY_RUN="$DRY_RUN" \
"$SCRIPT_DIR/run_work8_top_models_feature_sweep.sh"

