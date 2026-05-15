#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURABLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="${1:-/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/local_runs/3dunet_configurable_gridfix_label_sweep}"
PYTHON_BIN="${PYTHON_BIN:-$CONFIGURABLE_DIR/../.venv/bin/python}"

CONFIGS=(
  "config/local/gridfix_smoke_box72_electrostatic_shape_dataset_pos5.yml"
  "config/local/gridfix_smoke_box72_electrostatic_shape_dataset_pos10.yml"
  "config/local/gridfix_smoke_box72_electrostatic_shape_calculated_pos5.yml"
  "config/local/gridfix_smoke_box72_electrostatic_shape_calculated_pos10.yml"
  "config/local/gridfix_smoke_box72_oldbest_dataset_label_pos1.yml"
  "config/local/gridfix_smoke_box72_oldbest_calculated_label_pos1.yml"
  "config/local/gridfix_smoke_box72_oldbest_dataset_shape_pos1.yml"
  "config/local/gridfix_smoke_box72_oldbest_dataset_electrostatic_pos1.yml"
  "config/local/gridfix_smoke_box72_oldbest_dataset_electrostatic_shape_hydrophobic_pos1.yml"
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
