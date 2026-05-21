#!/bin/bash
#SBATCH --job-name=sheaf_synth_smoke
#SBATCH --output=/home/3393519/ts-gnn/logs/synthetic_sheaf_%j.out
#SBATCH --error=/home/3393519/ts-gnn/logs/synthetic_sheaf_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=stud
#SBATCH --account=3393519
#SBATCH --qos=stud

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/3393519/ts-gnn}"
RUN_ID="synthetic-sheaf-${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_DIR}/results/synthetic_sheaf/${RUN_ID}"
OUT_JSON="${OUT_DIR}/synthetic_sheaf_manifest.json"
RUNTIME_JSON="${OUT_DIR}/runtime_manifest.json"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python scripts/run_synthetic_sheaf_backtest.py \
  --out "$OUT_JSON" \
  --run-id "$RUN_ID" \
  --num-nodes 8 \
  --stalk-dim 2 \
  --num-steps 8 \
  --tau 0.06 \
  --seed 41 \
  --train-steps 4 \
  --random-seed 123

python - <<PY
from pathlib import Path
from sheaf_ts.hpc.manifest import build_runtime_manifest, write_runtime_manifest

repo = Path("$REPO_DIR")
out_json = Path("$OUT_JSON")
runtime_json = Path("$RUNTIME_JSON")
manifest = build_runtime_manifest(
    run_id="$RUN_ID",
    command=[
        "python",
        "scripts/run_synthetic_sheaf_backtest.py",
        "--out",
        "$OUT_JSON",
        "--run-id",
        "$RUN_ID",
    ],
    output_paths=[out_json],
    cwd=repo,
    extra={"kind": "synthetic_sheaf_slurm_smoke"},
)
write_runtime_manifest(manifest, runtime_json)
print("runtime_manifest", runtime_json)
PY
