#!/usr/bin/env bash
#SBATCH --job-name=deep-apbs-config-array
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00

set -euo pipefail

if [[ -z "${CONFIG_LIST:-}" ]]; then
  echo "CONFIG_LIST is required"
  exit 1
fi
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  echo "OUTPUT_ROOT is required"
  exit 1
fi

REPO_DIR="${REPO_DIR:-/homes/tevfik/PHD/3dunet-apbs/3dunet_configurable}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-ResNet3D4L}"
BASE_FEATURES="${BASE_FEATURES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "$REPO_DIR"

if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
fi

task_id="${SLURM_ARRAY_TASK_ID:-1}"
config_path="$(sed -n "${task_id}p" "$CONFIG_LIST")"

if [[ -z "$config_path" ]]; then
  echo "No config found at task index $task_id in $CONFIG_LIST"
  exit 1
fi

run_name="$(basename "$config_path" .yml)"
run_dir="$OUTPUT_ROOT/$run_name"
log_path="$run_dir/log/training.log"
final_model_path="$run_dir/weights/${MODEL}_final_model.pth"

echo "Deep-APBS config-array training"
echo "Host: $(hostname)"
echo "Task: $task_id"
echo "Repo: $REPO_DIR"
echo "Python: $PYTHON_BIN"
echo "Config list: $CONFIG_LIST"
echo "Config: $config_path"
echo "Output root: $OUTPUT_ROOT"
echo "Run: $run_name"
echo "Training log: $log_path"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Num workers: $NUM_WORKERS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

if [[ "${SKIP_COMPLETED:-1}" == "1" && -f "$final_model_path" ]]; then
  echo "Skipping completed run: $run_name"
  exit 0
fi

if [[ "${CLEAN_INCOMPLETE:-1}" == "1" && -d "$run_dir" && ! -f "$final_model_path" ]]; then
  echo "Cleaning incomplete run directory before restart: $run_dir"
  rm -rf "$run_dir"
fi

PYTHONUNBUFFERED=1 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" main.py \
  --config "$config_path" \
  --model "$MODEL" \
  --base_features "$BASE_FEATURES" \
  --num_workers "$NUM_WORKERS" \
  --base_model_output_dir "$OUTPUT_ROOT"
