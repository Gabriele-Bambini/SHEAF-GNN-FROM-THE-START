from __future__ import annotations

from pathlib import Path

from hpc_ssh import DEFAULT_REMOTE_ROOT, run_print, scp_to_remote


if __name__ == "__main__":
    local = Path(__file__).with_name("submit.sbatch")
    run_print(f"mkdir -p {DEFAULT_REMOTE_ROOT}/hpc", check=True)
    scp_to_remote([local], f"{DEFAULT_REMOTE_ROOT}/hpc/", check=True)
    run_print(f"cd {DEFAULT_REMOTE_ROOT} && sed -i 's/\r$//' hpc/submit.sbatch && sbatch hpc/submit.sbatch", check=True)
