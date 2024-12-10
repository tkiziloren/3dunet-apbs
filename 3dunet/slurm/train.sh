#!/bin/bash
#SBATCH --job-name=pdb-training        # İş ismi
#SBATCH --output=logs/%x_%j.log        # Çıkış dosyası (logs dizinine kaydeder)
#SBATCH --error=logs/%x_%j.err         # Hata dosyası
#SBATCH --partition=gpu                # GPU için uygun partition
#SBATCH --nodes=1                      # 1 node kullan
#SBATCH --gres=gpu:4                   # 4 GPU tahsis et
#SBATCH --ntasks=1                     # Tek görev
#SBATCH --cpus-per-task=16             # 16 CPU çekirdeği
#SBATCH --mem=64G                      # 64 GB RAM
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL                # Job bitiş ve hata durumlarında mail gönder
#SBATCH --mail-user=tevfik             # Mail adresiniz

# Modül yükleme (gerekiyorsa)
module load python/3.10.10               # Örnek bir Python modülü, cluster'a bağlı olarak değişebilir

# Sanal ortamı aktif etme
source ~/PHD/3dunet-apbs/venv/bin/activate

# Ana kodu çalıştırma
python main.py --config config/codon/pdbbind.yml