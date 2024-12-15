#!/bin/bash
#SBATCH --job-name=3d-unet
#SBATCH --gres=gpu:v100:${1:-1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tevfik
# Parametreler
LOG_PREFIX=${2:-job_output}  # Log dosyası prefix'i
CONFIG_PATH=${3:-config/config.yml}  # Config dosyası

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(readlink -f "$SCRIPT_DIR/..")  # Ana dizin
cd "$ROOT_DIR"

LOG_DIR=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs
STDOUT_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.out"
STDERR_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.err"

module load python/3.10.10
source ~/PHD/3dunet-apbs/venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config "${CONFIG_PATH}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"
#### RUN LIKE THIS: sbatch train.sh 2 pdbbind config/codon/pdbbind.yml