#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="${1:-/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/local_runs/3dunet_configurable_box161_loss_calibration}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"

CONFIGS=(
  "config/local/box161_loss_calibration/box161_compact_pos5_long.yml"
  "config/local/box161_loss_calibration/box161_compact_pos10_long.yml"
  "config/local/box161_loss_calibration/box161_compact_pos25_long.yml"
)

cd "$CONFIGURABLE_DIR"
mkdir -p "$OUTPUT_ROOT"

echo "Output root: $OUTPUT_ROOT"
echo "Python: $PYTHON_BIN"

for config_path in "${CONFIGS[@]}"; do
  run_name="$(basename "$config_path" .yml)"
  log_path="$OUTPUT_ROOT/$run_name/log/training.log"
  echo
  echo "=== Running $run_name ==="
  echo "Config: $config_path"
  echo "Training log: $log_path"
  echo "Tail with: tail -f $log_path"

  PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON_BIN" main.py \
    --config "$config_path" \
    --model UNet3D4L \
    --base_features 32 \
    --num_workers 0 \
    --base_model_output_dir "$OUTPUT_ROOT"
done

echo
echo "All runs finished. Summarize with:"
echo "$PYTHON_BIN scripts/summarize_gridfix_runs.py --output-root $OUTPUT_ROOT"
