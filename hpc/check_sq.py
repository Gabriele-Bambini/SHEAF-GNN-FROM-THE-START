from __future__ import annotations

from hpc_ssh import run_print


if __name__ == "__main__":
    run_print("squeue --me 2>/dev/null || squeue -u $(whoami)", check=False)
