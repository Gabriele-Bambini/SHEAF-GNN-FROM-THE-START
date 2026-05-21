#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/ts-gnn}"
cd "$REPO_DIR"

module purge || true
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tsgnn

python - <<'PY'
import importlib.util

if importlib.util.find_spec("optuna") is None:
    raise SystemExit("optuna is missing from the locked tsgnn environment")

import optuna

print("optuna", optuna.__version__)
PY
