from __future__ import annotations

import argparse

from hpc_ssh import run_print


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="Required before cancelling all jobs.")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to cancel jobs without --confirm")
    run_print("echo 'before:' && (squeue --me 2>/dev/null || squeue -u $(whoami)); scancel -u $(whoami); echo 'after:' && (squeue --me 2>/dev/null || squeue -u $(whoami))", check=False)
