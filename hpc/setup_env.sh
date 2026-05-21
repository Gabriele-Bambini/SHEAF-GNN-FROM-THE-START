#!/bin/bash
# Run ONCE on the HPC login node to create the conda environment.
# Usage: bash hpc/setup_env.sh
set -euo pipefail

CONDA_ENV="tsgnn"
REPO="/home/3393519/ts-gnn"

echo "=== Loading modules ==="
module load miniconda3
module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda env '$CONDA_ENV' (skip if exists) ==="
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  Env already exists, skipping create."
else
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

conda activate "$CONDA_ENV"

echo "=== PyTorch 2.2 + CUDA 12.1 ==="
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== PyG scatter/sparse (pinned to torch 2.2 cu121) ==="
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric==2.5.0

echo "=== Bio stack ==="
pip install scanpy==1.10.0 anndata==0.10.5 scvelo==0.3.2 \
    harmonypy==0.0.9 scrublet==0.2.3 decoupler==1.6.0 \
    igraph==0.11.6 leidenalg==0.10.2

echo "=== ESM-2 ==="
pip install fair-esm==2.0.0

echo "=== Data / utils ==="
pip install GEOparse==2.0.4 requests==2.31.0 \
    pandas==2.2.0 numpy==1.26.4 scipy==1.12.0 \
    matplotlib==3.8.3 seaborn==0.13.2 networkx==3.2.1 plotly==5.18.0 \
    wandb==0.16.3 hydra-core==1.3.2 omegaconf==2.3.0 PyYAML==6.0.1 \
    tqdm==4.66.2 mygene==3.2.2 scikit-learn==1.4.1.post1 pyarrow==15.0.2

echo "=== Install tsgnn package in editable mode ==="
cd "$REPO"
pip install -e .

echo ""
echo "=== Setup complete! Test with: ==="
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
