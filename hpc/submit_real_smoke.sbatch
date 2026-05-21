#!/bin/bash
#SBATCH --job-name=sheaf_real_smoke
#SBATCH --output=/home/3393519/ts-gnn/logs/real_smoke_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/real_smoke_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
RUN_ID="real-smoke-${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_DIR}/results/real_smoke/${RUN_ID}"
OUT_JSON="${OUT_DIR}/real_smoke_manifest.json"
RUNTIME_JSON="${OUT_DIR}/runtime_manifest.json"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python scripts/run_real_data_smoke.py \
  --out "$OUT_JSON" \
  --runtime-manifest "$RUNTIME_JSON" \
  --run-id "$RUN_ID" \
  --allele WT \
  --train-steps 6 \
  --random-seed 123
