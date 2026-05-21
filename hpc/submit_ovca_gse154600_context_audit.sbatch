#!/bin/bash
#SBATCH --job-name=ovca_gse154600_audit
#SBATCH --output=/home/3393519/ts-gnn/logs/ovca_gse154600_audit_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/ovca_gse154600_audit_%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
RUN_ID="ovca-gse154600-audit-${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_DIR}/results/ovarian_context_audit/${RUN_ID}"
OUT_JSON="${OUT_DIR}/ovarian_context_graph_audit_${SLURM_JOB_ID:-local}.json"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

export TSGNN_DATA_DIR="${TSGNN_DATA_DIR:-${REPO_DIR}/data}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/FROM THE START/src:${PYTHONPATH:-}"

python - <<'PY'
from tsgnn.data.ovarian_loader import download_ovarian_hgsoc_gse154600, ovarian_hgsoc_gse154600_available

raw_dir = download_ovarian_hgsoc_gse154600()
print(f"[ovca_gse154600] raw_dir={raw_dir}")
print(f"[ovca_gse154600] available={ovarian_hgsoc_gse154600_available()}")
PY

python "FROM THE START/scripts/audit_ovarian_context_graph.py" \
  --dataset ovarian_hgsoc_gse154600 \
  --out "$OUT_JSON"

echo "[ovca_gse154600] wrote: $OUT_JSON"
