from __future__ import annotations

import argparse

from hpc_ssh import DEFAULT_REMOTE_ROOT, run_print

DEFAULT_RUN = "results/runs/20260424-220334_nature_seed42_full_train-zeroshot-swap-bulk-ablate"
FILES = ["nature_run_summary.json", "benchmarks.json", "allele_swap.json", "holdout.json", "manifest.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()
    for name in FILES:
        print(f"\n{'=' * 60}\nFILE: {name}\n{'=' * 60}")
        run_print(f"cat {DEFAULT_REMOTE_ROOT}/{args.run}/{name} 2>/dev/null || true", check=False)
