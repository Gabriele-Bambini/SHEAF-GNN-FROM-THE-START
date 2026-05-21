#!/bin/bash
#SBATCH --job-name=ovca_gse154600_gate
#SBATCH --output=/home/3393519/ts-gnn/logs/ovca_gse154600_gate_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/ovca_gse154600_gate_%j.err
#SBATCH --time=01:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
FTS_DIR="${REPO_DIR}/FROM THE START"
RUN_ID="ovca-gse154600-context-gate-${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_DIR}/results/ovarian_context_gate/${RUN_ID}"
OUT_JSON="${OUT_DIR}/ovarian_context_gate_${SLURM_JOB_ID:-local}.json"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

export TSGNN_DATA_DIR="${TSGNN_DATA_DIR:-${REPO_DIR}/data}"
export PYTHONPATH="${FTS_DIR}/src:${REPO_DIR}/src:${PYTHONPATH:-}"

python - <<'PY'
from tsgnn.data.ovarian_loader import download_ovarian_hgsoc_gse154600, ovarian_hgsoc_gse154600_available

raw_dir = download_ovarian_hgsoc_gse154600()
print(f"[ovca_gse154600] raw_dir={raw_dir}")
print(f"[ovca_gse154600] available={ovarian_hgsoc_gse154600_available()}")
PY

python "${FTS_DIR}/scripts/evaluate_ovarian_context_gate.py" \
  --target-dataset ovarian_hgsoc_gse154600 \
  --out "$OUT_JSON"

echo "[ovca_gse154600_gate] wrote: $OUT_JSON"

