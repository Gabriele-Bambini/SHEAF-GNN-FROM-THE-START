#!/bin/bash
# HPC Bootstrap Script for TS-GNN
# Run this on the HPC login node once: bash hpc/bootstrap.sh

set -e

echo ">>> Loading miniconda3..."
module load miniconda3

CONDA_ENV="tsgnn"

# Check if environment exists
if conda info --envs | grep -q "^${CONDA_ENV} "; then
    echo ">>> Environment '${CONDA_ENV}' already exists. Updating..."
else
    echo ">>> Creating conda environment '${CONDA_ENV}' with Python 3.11..."
    conda create -n "${CONDA_ENV}" python=3.11 -y
fi

# Initialize shell for conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo ">>> Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing other dependencies from requirements.txt..."
pip install -r requirements.txt
pip install igraph==0.11.6 leidenalg==0.10.2 scikit-learn==1.4.1.post1 pyarrow==15.0.2

echo ">>> Installing project in editable mode..."
pip install -e .

echo ">>> Setup complete."
