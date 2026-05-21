#!/bin/bash
#SBATCH --job-name=tsgnn_v2_train
#SBATCH --output=/home/3393519/ts-gnn/logs/train_v2_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/train_v2_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

echo "=== TSGNN-v2 Full Training job $SLURM_JOB_ID ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /home/3393519/ts-gnn
mkdir -p logs results

module purge
module load cuda/12.1
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python -c "import torchdiffeq" >/dev/null

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

CONFIG="${TSGNN_CONFIG:-configs/nature_v2.yaml}"
echo "Config: $CONFIG"

python scripts/run_nature.py --config "$CONFIG" \
    2>&1 | tee logs/train_v2_${SLURM_JOB_ID}_stdout.txt

echo "End: $(date)"
