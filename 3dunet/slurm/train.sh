#!/bin/bash

# Dinamik parametreler
LOG_PREFIX=${1:-job_output}
CONFIG_PATH=${2:-config/config.yml}
MODEL_CLASS=${3:-UNet3D4L}  # 3. parametre: model sınıfı adı (varsayılan: UNet3D4L)
GPU_COUNT=${4:-1}
GPU_TYPE=${5:-h200}       # 5. parametre: GPU tipi (örn: h200, a100, l40s, v100)
CPUS_PER_TASK=${6:-16}    # 6. parametre: CPU sayısı (default: 16)
BASE_FEATURES=${7:-64}    # 7. parametre: temel özellik sayısı (varsayılan: 64)
MEMORY_GB=${8:-128}       # 8. parametre: bellek boyutu (varsayılan: 64GB)


LOG_DIR=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs
STDOUT_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.out"
STDERR_LOG="${LOG_DIR}/${LOG_PREFIX}_%j.err"

# Slurm job’u komut satırı parametreleriyle çalıştır
sbatch --job-name=3d-unet \
       --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
       --ntasks=1 \
       --cpus-per-task=${CPUS_PER_TASK} \
       --mem=${MEMORY_GB}G \
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


