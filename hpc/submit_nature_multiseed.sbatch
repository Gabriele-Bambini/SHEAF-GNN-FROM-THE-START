#!/bin/bash
#SBATCH --job-name=tsgnn_nature_3s
#SBATCH --output=/home/3393519/ts-gnn/logs/nature_ms_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/nature_ms_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

echo "=== Nature multi-seed (42,123,456) job $SLURM_JOB_ID ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /home/3393519/ts-gnn
mkdir -p logs results

module purge
module load cuda/12.1
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python -c "import torchdiffeq" 2>/dev/null || pip install --user torchdiffeq
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python scripts/run_nature.py --config configs/nature.yaml --phase train holdout swap bulk ablate --seeds 42,123,456

echo "End: $(date)"
