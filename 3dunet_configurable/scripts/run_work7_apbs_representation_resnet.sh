#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/work7_apbs_representation_resnet3d4l_fold1_250epoch_thr040}"
MODEL="${MODEL:-ResNet3D4L}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-250}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"
CUTOFF_VARIANTS="${CUTOFF_VARIANTS:-apbs_clip5_minmax,apbs_clip10_minmax,apbs_clip20_minmax,apbs_no_cutoff_current,apbs_full_minmax,apbs_full_signed,apbs_clip20_signed,apbs_posneg_clip20}"

mkdir -p "$OUTPUT_ROOT"

echo "Work7 APBS representation ablation"
echo "Output root: $OUTPUT_ROOT"
echo "Model: $MODEL"
echo "Variants: $CUTOFF_VARIANTS"

OUTPUT_ROOT="$OUTPUT_ROOT" \
CONFIG_DIR="$OUTPUT_ROOT/generated_configs" \
MODEL="$MODEL" \
FEATURE_SET=apbs_only \
CUTOFF_VARIANTS="$CUTOFF_VARIANTS" \
FOLDS="$FOLDS" \
EPOCHS="$EPOCHS" \
EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
VALIDATION_THRESHOLD="$VALIDATION_THRESHOLD" \
BASE_FEATURES="$BASE_FEATURES" \
NUM_WORKERS="$NUM_WORKERS" \
SKIP_COMPLETED="$SKIP_COMPLETED" \
CLEAN_INCOMPLETE="$CLEAN_INCOMPLETE" \
DRY_RUN="$DRY_RUN" \
"$SCRIPT_DIR/run_apbs_cutoff_sweep.sh"
