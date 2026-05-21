from __future__ import annotations

from hpc_ssh import run_print


if __name__ == "__main__":
    run_print("srun echo hello", check=False)
