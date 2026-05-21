#!/bin/bash
#SBATCH --job-name=tsgnn_smoke
#SBATCH --output=/home/3393519/ts-gnn/logs/smoke_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

echo "=== smoke_test job $SLURM_JOB_ID ==="
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

python hpc/smoke_test.py

echo "End: $(date)"
