#!/bin/bash
#SBATCH --job-name=sheaf_real_hpo_mega
#SBATCH --output=/home/3393519/ts-gnn/logs/real_hpo_mega_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/real_hpo_mega_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
RUN_ID="real-hpo-mega-${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_DIR}/results/real_hpo_mega/${RUN_ID}"
OUT_JSON="${OUT_DIR}/real_hpo_manifest.json"
RUNTIME_JSON="${OUT_DIR}/runtime_manifest.json"
STUDY_DB="${OUT_DIR}/optuna_study.db"
N_TRIALS="${N_TRIALS:-128}"
HPO_TIMEOUT="${HPO_TIMEOUT:-14400}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python -c "import optuna" >/dev/null

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python scripts/run_real_hpo_smoke.py \
  --out "$OUT_JSON" \
  --runtime-manifest "$RUNTIME_JSON" \
  --study-db "$STUDY_DB" \
  --artifact-dir "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --n-trials "$N_TRIALS" \
  --n-folds 2 \
  --timeout "$HPO_TIMEOUT" \
  --seed 42 \
  --allele WT
