#!/bin/bash
#SBATCH --job-name=tsgnn_probe_alpha
#SBATCH --output=/home/3393519/ts-gnn/logs/probe_alpha_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/probe_alpha_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

echo "=== probe_alpha (full stack vs baseline) job $SLURM_JOB_ID ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /home/3393519/ts-gnn
mkdir -p logs

module purge
module load cuda/12.1
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python -c "import torchdiffeq" >/dev/null

# Verify GPU visible
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

CONFIG="${TSGNN_CONFIG:-configs/nature.yaml}"
python hpc/probe_alpha.py --config "$CONFIG"

echo "End: $(date)"
