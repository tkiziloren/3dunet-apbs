#!/bin/bash

#=============================================================================
# CONFIGURABLE FEATURE VERSION - Feature sayısı config'den otomatik okunur
#=============================================================================
# USAGE:
#   bash train_new.sh JOB_NAME CONFIG_FILE MODEL_CLASS [GPU_COUNT] [GPU_TYPE] [CPUS] [BASE_FEATURES] [MEMORY_GB]
#
# REQUIRED:
#   JOB_NAME      : İş adı ve log dosyası prefix'i
#   CONFIG_FILE   : Config dosyası yolu (config/codon_with_different_selections/xxx.yml)
#   MODEL_CLASS   : Model sınıfı (UNet3D4L, UNet3D5L, ConvNeXt3D, vb.)
#
# OPTIONAL (defaults):
#   GPU_COUNT     : GPU sayısı (default: 1)
#   GPU_TYPE      : GPU tipi - h200, a100, l40s, v100 (default: h200)
#   CPUS          : CPU sayısı (default: 16)
#   BASE_FEATURES : Model base feature sayısı (default: 64)
#   MEMORY_GB     : Bellek GB (default: 128)
#
# EXAMPLES:
#   bash train_new.sh electrostatic_shape_dataset config/codon_with_different_selections/electrostatic_shape_label_dataset_binding_site.yml UNet3D4L
#   bash train_new.sh baseline_calculated config/codon_with_different_selections/baseline_label_calculated_binding_site.yml UNet3D5L 1 h200 16 64 128
#   bash train_new.sh donor_acceptor_dataset config/codon_with_different_selections/electrostatic_donor_acceptor_label_dataset_binding_site.yml ConvNeXt3D
#=============================================================================

# Parametreler
JOB_NAME=${1:-job_output}
CONFIG_PATH=${2:-config/config.yml}
MODEL_CLASS=${3:-UNet3D4L}
GPU_COUNT=${4:-1}
GPU_TYPE=${5:-h200}
CPUS_PER_TASK=${6:-16}
BASE_FEATURES=${7:-64}
MEMORY_GB=${8:-128}

LOG_DIR="/hps/nobackup/arl/chembl/tevfik/3dunet-apbs/logs/${JOB_NAME}"
WEIGHTS_DIR="/hps/nobackup/arl/chembl/tevfik/3dunet-apbs/output/${JOB_NAME}"
STDOUT_LOG="${LOG_DIR}/${JOB_NAME}_%j.out"
STDERR_LOG="${LOG_DIR}/${JOB_NAME}_%j.err"

echo "=========================================="
echo "Submitting Training Job"
echo "=========================================="
echo "Job Name     : ${JOB_NAME}"
echo "Config       : ${CONFIG_PATH}"
echo "Model        : ${MODEL_CLASS}"
echo "GPU          : ${GPU_COUNT}x ${GPU_TYPE}"
echo "CPUs         : ${CPUS_PER_TASK}"
echo "Base Features: ${BASE_FEATURES}"
echo "Memory       : ${MEMORY_GB}GB"
echo "Log Dir      : ${LOG_DIR}"
echo "Weights Dir  : ${WEIGHTS_DIR}"
echo "=========================================="

# Slurm job'u komut satırı parametreleriyle çalıştır
sbatch --job-name=${JOB_NAME} \
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
               cd ~/PHD/3dunet-apbs/3dunet_configurable && \
               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
               python main.py \
                --config ${CONFIG_PATH} \
                --model ${MODEL_CLASS} \
                --base_features ${BASE_FEATURES} \
                --num_workers ${CPUS_PER_TASK} \
                --base_model_output_dir ${WEIGHTS_DIR}"

echo "Job submitted!"
