from __future__ import annotations

from hpc_ssh import DEFAULT_REMOTE_ROOT, run_print


if __name__ == "__main__":
    run_print(f"cd {DEFAULT_REMOTE_ROOT} && sbatch hpc/submit.sbatch", check=True)
