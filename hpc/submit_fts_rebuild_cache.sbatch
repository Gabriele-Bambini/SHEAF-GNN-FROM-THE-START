#!/bin/bash
#SBATCH --job-name=fts_rebuild_cache
#SBATCH --output=/home/3393519/ts-gnn/logs/fts_rebuild_cache_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/fts_rebuild_cache_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
FTS_DIR="${REPO_DIR}/FROM THE START"
export TSGNN_DATA_DIR="${TSGNN_DATA_DIR:-${REPO_DIR}/data}"
RUN_ID="fts-rebuild-cache-${SLURM_JOB_ID:-local}"
OUT_ROOT="${FTS_DIR}/results/hpc_rebuild_cache/${RUN_ID}"
CACHE_DIR="${OUT_ROOT}/trusted_cache"
OUT_JSON="${OUT_ROOT}/trusted_cache_manifest.json"
AUDIT_JSON="${OUT_ROOT}/trusted_cache_audit.json"

echo "=== FROM-THE-START trusted real-cache rebuild ${SLURM_JOB_ID:-local} ==="
echo "Node: ${SLURMD_NODENAME:-local}"
echo "TSGNN_DATA_DIR: $TSGNN_DATA_DIR"
echo "Start: $(date)"

cd "$FTS_DIR"
mkdir -p "$OUT_ROOT" "$CACHE_DIR" "${REPO_DIR}/logs"

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

export PYTHONPATH="${FTS_DIR}/src:${REPO_DIR}/src:${PYTHONPATH:-}"

python scripts/rebuild_real_cache.py \
  --out-dir "$CACHE_DIR" \
  --n-genes "${N_GENES:-500}" \
  --K "${K_BINS:-10}" \
  --k-neighbors "${K_NEIGHBORS:-8}" \
  --pseudotime-method "${PSEUDOTIME_METHOD:-dpt}" \
  --out "$OUT_JSON"

python scripts/audit_real_cache.py \
  --data-dir "$CACHE_DIR" \
  --out "$AUDIT_JSON"

echo "Manifest: $OUT_JSON"
echo "Audit: $AUDIT_JSON"
echo "End: $(date)"
