#!/usr/bin/env bash
# TS-GNN Google Colab setup script
# Usage: bash scripts/setup_colab.sh [--drive-path /content/drive/MyDrive/tsgnn]
#
# Installs all dependencies in the correct order for Colab.
# Run once per Colab session (Colab resets env on disconnect).

set -euo pipefail

DRIVE_PATH="${1:-/content/drive/MyDrive/tsgnn}"
PROJECT_DIR="$DRIVE_PATH/ts-gnn"

echo "========================================="
echo "  TS-GNN Colab Setup"
echo "  Project: $PROJECT_DIR"
echo "========================================="

# 1. Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# 2. Upgrade pip silently
pip install --upgrade pip -q

# 3. Install PyTorch (already present in Colab, just verify)
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"

# 4. Install PyG (must match torch version)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.','')[:3] if torch.cuda.is_available() else 'cpu')")
echo "Installing PyG for torch=$TORCH_VERSION cuda=$CUDA_VERSION..."
pip install torch-geometric==2.5.0 -q
pip install pyg-lib torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html" -q || \
    echo "Warning: pyg-lib/scatter/sparse not installed (CPU fallback OK for testing)"

# 5. Install biology stack
echo "Installing biology stack..."
pip install \
    scanpy==1.10.0 \
    scvelo==0.3.2 \
    anndata==0.10.5 \
    harmonypy==0.0.9 \
    scrublet==0.2.3 \
    decoupler==1.6.0 \
    fair-esm==2.0.0 \
    GEOparse==2.0.4 \
    -q

# scvi-tools (optional — gold-standard batch correction, Lopez 2018 NatMethods)
echo "Installing scvi-tools (optional, ~2 min)..."
pip install scvi-tools -q 2>&1 | tail -3 || echo "Warning: scvi-tools install failed — Harmony fallback will be used"

# 6. Install remaining deps
echo "Installing remaining dependencies..."
pip install \
    wandb==0.16.3 \
    tqdm==4.66.2 \
    pyyaml==6.0.1 \
    networkx==3.2.1 \
    optuna==3.5.0 \
    -q

# 7. Install project in editable mode
echo "Installing ts-gnn project..."
pip install -e "$PROJECT_DIR" --no-build-isolation -q

# 8. Verify imports
echo "Verifying installation..."
python -c "
import tsgnn
import torch
import scanpy
import decoupler
print(f'tsgnn OK: {tsgnn.__file__}')
print(f'torch: {torch.__version__}')
print(f'scanpy: {scanpy.__version__}')
print(f'decoupler: {decoupler.__version__}')
"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Next: python scripts/download_all_datasets.py"
echo "========================================="
