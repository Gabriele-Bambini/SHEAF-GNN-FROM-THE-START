#!/bin/bash
#SBATCH --job-name=tsgnn_ovarian_zs
#SBATCH --output=/home/3393519/ts-gnn/logs/nature_ovarian_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/nature_ovarian_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

echo "=== TS-GNN BRCA→Ovarian zero-shot job $SLURM_JOB_ID ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /home/3393519/ts-gnn
mkdir -p logs results

# Check ovarian data is present before starting GPU job
OVA_DIR="data/raw/ovarian"
OVA_10X="${OVA_DIR}/GSE146026_Izar_HGSOC_ascites_10x_log.tsv.gz"
if [[ ! -f "${OVA_10X}" ]]; then
    echo "ERROR: Ovarian data not found at ${OVA_10X}"
    echo "Run: bash hpc/download_ovarian_data.sh first"
    exit 1
fi
echo "Ovarian data: $(du -sh ${OVA_10X} | cut -f1)"

module purge
module load cuda/12.1
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python -c "import torchdiffeq" 2>/dev/null || pip install --user torchdiffeq
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# Run with ovarian config — zeroshot only (train+zeroshot phases, skip bulk/swap/ablate)
python scripts/run_nature.py \
    --config configs/nature_ovarian.yaml \
    --phase train zeroshot \
    --seeds 42,123,456

echo "End: $(date)"
