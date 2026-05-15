#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Long APBS-only cutoff sweep.
# Runs the same three APBS normalization variants as run_apbs_cutoff_sweep.sh:
#   1. [-10, +10] clipping
#   2. [-20, +20] clipping
#   3. no cutoff
#
# Baseline [-5, +5] remains Work 1 fold0 apbs_only.

FEATURE_SET="${FEATURE_SET:-apbs_only}" \
EPOCHS="${EPOCHS:-250}" \
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}" \
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.40}" \
FOLDS="${FOLDS:-0}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/tevfik/Sandbox/github/PHD/runs/apbs_cutoff_sweep_apbs_only_fold0_250epoch_thr040}" \
"$SCRIPT_DIR/run_apbs_cutoff_sweep.sh"
