#!/bin/bash
#SBATCH --job-name=3d-unet
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tevfik
# Parametreler
LOG_PREFIX=${1:-job_output}  # Log dosyası prefix'i
CONFIG_PATH=${2:-config/config.yml}  # Config dosyası


cd /homes/tevfik/PHD/3dunet-apbs/3dunet
LOG_DIR=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs
STDOUT_LOG="${LOG_DIR}/${LOG_PREFIX}_${SLURM_JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/${LOG_PREFIX}_${SLURM_JOB_ID}.err"

echo "Config Path: $CONFIG_PATH"
echo "Log Prefix: $LOG_PREFIX"
echo "Stdout Log File: $STDOUT_LOG"
echo "Stderr Log File: $STDERR_LOG"

module load python/3.10.10
source ~/PHD/3dunet-apbs/venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config "${CONFIG_PATH}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"
#### RUN LIKE THIS: sbatch train.sh 2 pdbbind config/codon/pdbbind.yml