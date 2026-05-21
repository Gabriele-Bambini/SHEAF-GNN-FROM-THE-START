"""Run shortcut-resistant real-data sheaf smoke validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sheaf_ts.experiments.real_data_smoke import run_real_data_smoke
from sheaf_ts.hpc.manifest import build_runtime_manifest, write_runtime_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-path", default="data/processed/temporal_sequences.pt")
    parser.add_argument("--grn-path", default="data/processed/grn_artifacts.pt")
    parser.add_argument("--out", required=True, help="Path to output smoke manifest JSON")
    parser.add_argument("--run-id", default="real-data-smoke")
    parser.add_argument("--runtime-manifest", default=None, help="Optional runtime manifest JSON path")
    parser.add_argument("--allele", default="WT")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--edge-weight-power", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--degree-regularizer", type=float, default=1e-3)
    parser.add_argument("--no-temporal-edge-weights", action="store_true")
    parser.add_argument("--primary-model", default="sheaf_residual", choices=["sheaf", "sheaf_residual"])
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if no-shortcut gates reject the run.")
    args = parser.parse_args()

    manifest = run_real_data_smoke(
        temporal_path=args.temporal_path,
        grn_path=args.grn_path,
        out_path=args.out,
        run_id=args.run_id,
        allele=args.allele,
        train_steps=args.train_steps,
        random_seed=args.random_seed,
        edge_weight_power=args.edge_weight_power,
        ridge=args.ridge,
        degree_regularizer=args.degree_regularizer,
        use_temporal_edge_weights=not args.no_temporal_edge_weights,
        primary_model=args.primary_model,
        strict=args.strict,
    )
    verdict = manifest["gate_verdict"]
    primary = manifest["primary_model"]
    sheaf = manifest["result"]["model_metrics"][primary]["r_delta_perp_no_self"]
    controls = {
        name: metrics["r_delta_perp_no_self"]
        for name, metrics in manifest["result"]["baseline_metrics"].items()
    }
    print(
        "real_data_smoke",
        verdict["status"],
        f"{primary}_r_delta_perp_no_self",
        f"{sheaf:.6f}",
        "best_control",
        f"{max(controls.values()):.6f}",
        "manifest",
        args.out,
    )

    if args.runtime_manifest:
        runtime = build_runtime_manifest(
            run_id=args.run_id,
            command=sys.argv,
            output_paths=[Path(args.out)],
            cwd=ROOT,
            extra={"kind": "real_data_smoke_cli", "gate_status": verdict["status"]},
        )
        write_runtime_manifest(runtime, args.runtime_manifest)
        print("runtime_manifest", args.runtime_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
