#!/bin/bash
# Slurm direktiflerini script dışından sağlamak için yorum olarak bırakıyoruz.
# #SBATCH --job-name=3d-unet
# #SBATCH --gres=gpu:v100:1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=64G
# #SBATCH --time=7-00:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=tevfik

# Dinamik parametreler
LOG_PREFIX=${1:-job_output}
CONFIG_PATH=${2:-config/config.yml}
GPU_COUNT=${3:-1}

LOG_DIR=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs
STDOUT_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.out"
STDERR_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.err"

# Slurm job’u komut satırı parametreleriyle çalıştır
sbatch --job-name=3d-unet \
       --gres=gpu:v100:${GPU_COUNT} \
       --ntasks=1 \
       --cpus-per-task=16 \
       --mem=64G \
       --time=7-00:00:00 \
       --mail-type=ALL \
       --mail-user=tevfik \
       --output="${STDOUT_LOG}" \
       --error="${STDERR_LOG}" \
       --wrap="module load python/3.10.10 && \
               source ~/PHD/3dunet-apbs/venv/bin/activate && \
               cd ~/PHD/3dunet-apbs/3dunet && \
               python main.py --config ${CONFIG_PATH}"