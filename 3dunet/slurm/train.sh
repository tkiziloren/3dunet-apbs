#!/bin/bash
#SBATCH --job-name=3d-unet             # İş ismi
#SBATCH --output=logs/%x_%j.log        # Çıkış dosyası (logs dizinine kaydeder)
#SBATCH --error=logs/%x_%j.err         # Hata dosyası
#SBATCH --gres=gpu:4                   # 4 GPU tahsis et
#SBATCH --ntasks=1                     # Tek görev
#SBATCH --cpus-per-task=16             # 16 CPU çekirdeği
#SBATCH --mem=64G                      # 64 GB RAM
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL                # Job bitiş ve hata durumlarında mail gönder
#SBATCH --mail-user=tevfik             # Mail adresiniz
#SBATCH --partition=datamover

# Modül yükleme (gerekiyorsa)
module load python/3.10.10

# Sanal ortamı aktif etme
source ~/PHD/3dunet-apbs/venv/bin/activate

cd ../

export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config config/codon/pdbbind.yml