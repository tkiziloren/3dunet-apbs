#!/bin/bash
#SBATCH --job-name=3d-unet
#SBATCH --output=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/%x_%j.log
#SBATCH --error=/homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/%x_%j.err
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=tevfik

# Modül yükleme (gerekiyorsa)
module load python/3.10.10

# Sanal ortamı aktif etme
source ~/PHD/3dunet-apbs/venv/bin/activate

cd ../

export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config config/codon/pdbbind.yml