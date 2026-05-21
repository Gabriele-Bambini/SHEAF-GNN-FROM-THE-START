import json

import pytest
import torch


def test_synthetic_sheaf_dataset_is_deterministic_and_well_shaped():
    from sheaf_ts.experiments.synthetic_sheaf import generate_sheaf_diffusion_dataset

    a = generate_sheaf_diffusion_dataset(
        num_nodes=7,
        stalk_dim=3,
        num_steps=6,
        tau=0.08,
        seed=11,
    )
    b = generate_sheaf_diffusion_dataset(
        num_nodes=7,
        stalk_dim=3,
        num_steps=6,
        tau=0.08,
        seed=11,
    )

    assert a.states.shape == (7, 7, 3)
    assert a.deltas.shape == (6, 7, 3)
    assert a.maps.shape == (a.edge_index.shape[1], 2, 3, 3)
    assert torch.allclose(a.states, b.states)
    assert torch.allclose(a.laplacian, b.laplacian)


def test_oracle_sheaf_beats_mean_drift_and_random_graph_controls():
    from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates
    from sheaf_ts.experiments.synthetic_sheaf import (
        backtest_oracle_sheaf,
        generate_sheaf_diffusion_dataset,
    )

    dataset = generate_sheaf_diffusion_dataset(
        num_nodes=9,
        stalk_dim=3,
        num_steps=10,
        tau=0.06,
        seed=13,
    )
    result = backtest_oracle_sheaf(dataset, train_steps=5, random_seed=17)

    assert result.model_metrics["r_delta_perp"] > 0.99
    assert result.control_metrics["mean_drift"] < 0.5
    assert result.control_metrics["random_graph"] < 0.8

    verdict = evaluate_scientific_gates(
        model_metric=result.model_metrics["r_delta_perp"],
        controls=result.control_metrics,
        required=("mean_drift", "random_graph"),
        thresholds=GateThresholds(primary_margin=0.2, mechanism_margin=0.1),
    )
    assert verdict["accepted"], verdict


def test_optuna_can_select_tau_on_synthetic_cv_smoke():
    pytest.importorskip("optuna")

    from sheaf_ts.eval.cross_validation import temporal_blocked_splits
    from sheaf_ts.experiments.synthetic_sheaf import (
        generate_sheaf_diffusion_dataset,
        tune_tau_with_optuna,
    )

    dataset = generate_sheaf_diffusion_dataset(
        num_nodes=8,
        stalk_dim=2,
        num_steps=9,
        tau=0.07,
        seed=21,
    )
    splits = tuple(temporal_blocked_splits(n_bins=dataset.deltas.shape[0], min_train_bins=3, horizon=2))
    result = tune_tau_with_optuna(
        dataset,
        splits=splits,
        candidate_taus=(0.02, 0.07, 0.12),
        n_trials=3,
        seed=5,
    )

    assert result["best_params"]["tau"] == 0.07
    assert result["best_value"] > 0.99


def test_synthetic_manifest_is_json_serializable():
    from sheaf_ts.experiments.synthetic_sheaf import (
        backtest_oracle_sheaf,
        generate_sheaf_diffusion_dataset,
        synthetic_manifest,
    )

    dataset = generate_sheaf_diffusion_dataset(
        num_nodes=6,
        stalk_dim=2,
        num_steps=5,
        tau=0.05,
        seed=31,
    )
    result = backtest_oracle_sheaf(dataset, train_steps=3, random_seed=2)
    manifest = synthetic_manifest(dataset, result, run_id="synthetic-smoke")
    encoded = json.dumps(manifest, sort_keys=True)

    assert "synthetic-smoke" in encoded
    assert manifest["dataset"]["num_nodes"] == 6
    assert manifest["result"]["accepted_controls"]["mean_drift_margin"] > 0.0


def test_synthetic_runner_writes_manifest_with_strict_gate(tmp_path):
    from sheaf_ts.experiments.synthetic_runner import run_synthetic_backtest

    out_path = tmp_path / "manifest.json"
    manifest = run_synthetic_backtest(
        out_path=out_path,
        run_id="pytest-synthetic-runner",
        num_nodes=8,
        stalk_dim=2,
        num_steps=8,
        tau=0.06,
        seed=41,
        train_steps=4,
        strict=True,
    )

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["run_id"] == "pytest-synthetic-runner"
    assert manifest["gate_verdict"]["accepted"]
    assert loaded["gate_verdict"]["accepted"]
