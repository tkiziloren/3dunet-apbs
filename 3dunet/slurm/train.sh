#!/bin/bash
#SBATCH --job-name=3d-unet
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tevfik
#SBATCH --partition=datamover

# Modül yükleme (gerekiyorsa)
module load python/3.10.10

# Sanal ortamı aktif etme
source ~/PHD/3dunet-apbs/venv/bin/activate

cd ../

export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config config/codon/pdbbind.yml