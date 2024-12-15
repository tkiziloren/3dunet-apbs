#!/bin/bash
GPUS=${1:-1}
LOG_PREFIX=${2:-job_output}
CONFIG_PATH=${3:-config/config.yml}

cat << EOF > dynamic_train.sh
#!/bin/bash
#SBATCH --job-name=3d-unet
#SBATCH --gres=gpu:v100:${GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tevfik

module load python/3.10.10
source ~/PHD/3dunet-apbs/venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py --config "${CONFIG_PATH}" > /homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/${LOG_PREFIX}_%j.out 2> /homes/tevfik/PHD/3dunet-apbs/slurm_run_logs/${LOG_PREFIX}_%j.err
EOF

sbatch dynamic_train.sh