#!/bin/bash
#SBATCH --job-name=ts-gnn
#SBATCH --output=logs/tsgnn_%j.out
#SBATCH --error=logs/tsgnn_%j.err

# Account directives requested by Cluster Admin
#SBATCH --account=3393519
#SBATCH --partition=stud
#SBATCH --qos=stud

#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

# Load modules
module purge
module load cuda/12.1
module load miniconda3

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tsgnn

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
export TSGNN_DATA_DIR="${TSGNN_DATA_DIR:-${REPO_DIR}/data}"
export TSGNN_CLUSTER="${TSGNN_CLUSTER:-1}"
export TSGNN_STRICT_SCIENCE="${TSGNN_STRICT_SCIENCE:-1}"
ENTRYPOINT="${TSGNN_ENTRYPOINT:-scripts/run_nature.py}"
CONFIG_PATH="${TSGNN_CONFIG:-configs/nature.yaml}"
EXTRA_ARGS="${TSGNN_EXTRA_ARGS:-}"

# Ensure log directory exists
cd "$REPO_DIR"
mkdir -p logs results checkpoints "$TSGNN_DATA_DIR" "$TSGNN_DATA_DIR/processed"

echo "=== TS-GNN cluster job ==="
echo "REPO_DIR=$REPO_DIR"
echo "TSGNN_DATA_DIR=$TSGNN_DATA_DIR"
echo "ENTRYPOINT=$ENTRYPOINT"
echo "CONFIG_PATH=$CONFIG_PATH"

python hpc/check_data.py --data-dir "$TSGNN_DATA_DIR" --config "$CONFIG_PATH"
python "$ENTRYPOINT" --config "$CONFIG_PATH" $EXTRA_ARGS
