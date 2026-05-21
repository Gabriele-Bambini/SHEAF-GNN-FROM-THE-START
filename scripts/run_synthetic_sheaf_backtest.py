"""Run the clean synthetic sheaf backtest and write a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sheaf_ts.experiments.synthetic_runner import run_synthetic_backtest
from sheaf_ts.hpc.manifest import build_runtime_manifest, write_runtime_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Path to output manifest JSON")
    parser.add_argument("--run-id", default="synthetic-sheaf-smoke")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--stalk-dim", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--runtime-manifest", default=None, help="Optional runtime manifest JSON path")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    manifest = run_synthetic_backtest(
        out_path=args.out,
        run_id=args.run_id,
        num_nodes=args.num_nodes,
        stalk_dim=args.stalk_dim,
        num_steps=args.num_steps,
        tau=args.tau,
        seed=args.seed,
        train_steps=args.train_steps,
        random_seed=args.random_seed,
        strict=not args.no_strict,
    )
    print(
        "synthetic_sheaf_backtest",
        manifest["gate_verdict"]["status"],
        "manifest",
        args.out,
    )
    if args.runtime_manifest:
        runtime = build_runtime_manifest(
            run_id=args.run_id,
            command=sys.argv,
            output_paths=[Path(args.out)],
            cwd=ROOT,
            extra={"kind": "synthetic_sheaf_cli"},
        )
        write_runtime_manifest(runtime, args.runtime_manifest)
        print("runtime_manifest", args.runtime_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
