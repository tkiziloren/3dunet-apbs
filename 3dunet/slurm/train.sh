#!/bin/bash
GPUS=${1:-1}                           # GPU from first parameter, if not defined use 1
LOG_PREFIX=${2:-job_output}
CONFIG_PATH=${3:-config/config.yml}

#SBATCH --job-name=3d-unet
#SBATCH --output=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/${LOG_PREFIX}_%j.log
#SBATCH --error=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/${LOG_PREFIX}_%j.err
#SBATCH --gres=gpu:v100:${GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tevfik
module load python/3.10.10
source ~/PHD/3dunet-apbs/venv/bin/activate

# Script'in bulunduğu dizini belirle
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(readlink -f "$SCRIPT_DIR/..")  # Get parent folder
cd "$ROOT_DIR"  # Switch to main folder

export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config "${CONFIG_PATH}"
#### RUN LIKE THIS: sbatch train.sh 2 pdbbind config/codon/pdbbind.yml