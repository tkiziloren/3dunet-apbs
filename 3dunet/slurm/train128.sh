#!/bin/bash
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
       --mem=128G \
       --time=7-00:00:00 \
       --mail-type=ALL \
       --mail-user=tevfik \
       --output="${STDOUT_LOG}" \
       --error="${STDERR_LOG}" \
       --wrap="module load python/3.10.10 && \
               source ~/PHD/3dunet-apbs/venv/bin/activate && \
               cd ~/PHD/3dunet-apbs/3dunet && \
               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               python main.py --config ${CONFIG_PATH}"