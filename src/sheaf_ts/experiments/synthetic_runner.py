"""Local runner for the synthetic sheaf backtest."""

from __future__ import annotations

import json
from pathlib import Path

from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates
from sheaf_ts.experiments.synthetic_sheaf import (
    backtest_oracle_sheaf,
    generate_sheaf_diffusion_dataset,
    synthetic_manifest,
)

__all__ = ["run_synthetic_backtest"]


def run_synthetic_backtest(
    *,
    out_path: str | Path,
    run_id: str,
    num_nodes: int = 10,
    stalk_dim: int = 3,
    num_steps: int = 12,
    tau: float = 0.05,
    seed: int = 0,
    train_steps: int = 6,
    random_seed: int = 123,
    strict: bool = True,
) -> dict[str, object]:
    """Run synthetic backtest and write a JSON manifest."""
    dataset = generate_sheaf_diffusion_dataset(
        num_nodes=num_nodes,
        stalk_dim=stalk_dim,
        num_steps=num_steps,
        tau=tau,
        seed=seed,
    )
    result = backtest_oracle_sheaf(
        dataset,
        train_steps=train_steps,
        random_seed=random_seed,
    )
    manifest = synthetic_manifest(dataset, result, run_id=run_id)
    verdict = evaluate_scientific_gates(
        model_metric=result.model_metrics["r_delta_perp"],
        controls=result.control_metrics,
        required=("mean_drift", "random_graph"),
        thresholds=GateThresholds(primary_margin=0.2, mechanism_margin=0.1),
    )
    manifest["gate_verdict"] = verdict
    manifest["strict"] = bool(strict)

    if strict and not verdict["accepted"]:
        raise ValueError("synthetic backtest failed strict gates: " + verdict["status"])

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
