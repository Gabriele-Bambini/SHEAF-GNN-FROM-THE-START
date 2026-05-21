from __future__ import annotations

import shlex
import sys

from hpc_ssh import run_print


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python hpc/check_sq_start.py <job_id>")
    run_print(f"squeue --start -j {shlex.quote(sys.argv[1])}", check=False)
