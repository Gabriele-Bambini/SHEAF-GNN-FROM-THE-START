import json

import pytest
import torch


def test_experiment_result_rejects_nan_and_missing_required_controls():
    from sheaf_ts.eval.artifacts import ControlResult, ExperimentResult

    result = ExperimentResult(
        run_id="bad-run",
        level="L4",
        seed=42,
        fold=0,
        metrics={"r_delta_perp": float("nan")},
        controls={
            "mean_drift": ControlResult(name="mean_drift", metric=0.2),
        },
        metadata={"git_sha": "abc"},
    )

    with pytest.raises(ValueError, match="non-finite"):
        result.validate(required_controls=("mean_drift", "shuffled_esm"))


def test_experiment_result_json_roundtrip_is_plain_data():
    from sheaf_ts.eval.artifacts import ControlResult, ExperimentResult

    result = ExperimentResult(
        run_id="run-001",
        level="L2",
        seed=123,
        fold=2,
        metrics={"r_delta_perp": 0.42, "r_delta": 0.7},
        controls={
            "mean_drift": ControlResult(name="mean_drift", metric=0.31, delta=0.11),
            "random_graph": ControlResult(name="random_graph", metric=0.2, delta=0.22),
        },
        metadata={"config_hash": "cfg"},
    )

    payload = result.to_json_dict()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["run_id"] == "run-001"
    assert decoded["controls"]["mean_drift"]["delta"] == 0.11
    result.validate(required_controls=("mean_drift", "random_graph"))


def test_r_delta_perp_removes_train_only_mean_drift_shortcut():
    from sheaf_ts.eval.metrics import directional_metrics, fit_mean_drift

    torch.manual_seed(0)
    train_true = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )
    test_true = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, -1.0],
            [3.0, 3.0, 0.5],
        ],
        dtype=torch.float64,
    )
    mean_shortcut_pred = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )

    mean_drift = fit_mean_drift(train_true)
    metrics = directional_metrics(mean_shortcut_pred, test_true, mean_drift=mean_drift)

    assert metrics["r_delta"] > 0.7
    assert abs(metrics["r_delta_perp"]) < 1e-8


def test_scientific_gates_fail_when_controls_match_or_beat_model():
    from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates

    verdict = evaluate_scientific_gates(
        model_metric=0.50,
        controls={
            "mean_drift": 0.49,
            "random_graph": 0.50,
            "constant_graph": 0.20,
            "zero_esm": 0.10,
            "same_esm": 0.10,
            "shuffled_esm": 0.10,
            "shuffled_time": 0.10,
            "reversed_time": 0.10,
        },
        required=("mean_drift", "random_graph"),
        thresholds=GateThresholds(primary_margin=0.02, mechanism_margin=0.02),
    )

    assert not verdict["accepted"]
    assert "mean_drift" in verdict["failed_controls"]
    assert "random_graph" in verdict["failed_controls"]


def test_group_kfold_splits_are_patient_disjoint_and_deterministic():
    from sheaf_ts.eval.cross_validation import repeated_group_kfold

    groups = ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"]
    splits_a = list(repeated_group_kfold(groups, n_splits=3, n_repeats=2, seed=7))
    splits_b = list(repeated_group_kfold(groups, n_splits=3, n_repeats=2, seed=7))

    assert splits_a == splits_b
    assert len(splits_a) == 6
    for split in splits_a:
        train_groups = {groups[i] for i in split.train_idx}
        val_groups = {groups[i] for i in split.val_idx}
        assert train_groups.isdisjoint(val_groups)
        assert split.repeat in (0, 1)
        assert split.fold in (0, 1, 2)


def test_temporal_blocked_splits_never_train_on_future_bins():
    from sheaf_ts.eval.cross_validation import temporal_blocked_splits

    splits = list(temporal_blocked_splits(n_bins=8, min_train_bins=3, horizon=2))

    assert len(splits) >= 2
    for split in splits:
        assert max(split.train_idx) < min(split.val_idx)
        assert len(split.val_idx) == 2


def test_optuna_search_space_applies_to_fake_trial_without_importing_optuna():
    from sheaf_ts.hpo.optuna_runner import SearchSpace

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            assert name in {"lr", "weight_decay"}
            return low if log else high

        def suggest_categorical(self, name, choices):
            return choices[-1]

        def suggest_int(self, name, low, high):
            return high

    space = SearchSpace(
        floats={"lr": (1e-4, 1e-2, True), "weight_decay": (0.0, 1e-3, False)},
        categoricals={"stalk_dim": (2, 4, 8)},
        integers={"layers": (1, 3)},
    )
    params = space.suggest(FakeTrial())

    assert params == {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "stalk_dim": 8,
        "layers": 3,
    }


def test_optuna_cv_requires_nonempty_splits_before_importing_optuna():
    from sheaf_ts.hpo.optuna_runner import SearchSpace, run_optuna_cv

    with pytest.raises(ValueError, match="splits"):
        run_optuna_cv(
            objective=lambda params, split: 0.0,
            search_space=SearchSpace(),
            splits=[],
            n_trials=1,
        )


def test_optuna_cv_smoke_run_when_optuna_is_available():
    pytest.importorskip("optuna")

    from sheaf_ts.eval.cross_validation import CVSplit
    from sheaf_ts.hpo.optuna_runner import SearchSpace, run_optuna_cv

    splits = (
        CVSplit(train_idx=(0, 1), val_idx=(2,), fold=0),
        CVSplit(train_idx=(0, 2), val_idx=(1,), fold=1),
    )

    def objective(params, split):
        return float(params["stalk_dim"]) - 0.1 * split.fold

    result = run_optuna_cv(
        objective=objective,
        search_space=SearchSpace(categoricals={"stalk_dim": (2, 4, 8)}),
        splits=splits,
        n_trials=2,
        direction="maximize",
        study_name="sheaf_ts_pytest_smoke",
        seed=3,
    )

    assert result["best_params"]["stalk_dim"] in {2, 4, 8}
    assert result["trial_results"]
    assert all(len(trial.fold_scores) == 2 for trial in result["trial_results"])


def test_bootstrap_ci_and_bh_fdr_are_deterministic_and_valid():
    from sheaf_ts.eval.stats import benjamini_hochberg, bootstrap_mean_ci

    ci = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=200, seed=4)
    adjusted = benjamini_hochberg([0.001, 0.02, 0.20, 0.04])

    assert ci["mean"] == pytest.approx(0.3)
    assert ci["low"] <= ci["mean"] <= ci["high"]
    assert adjusted[0] <= adjusted[1] <= adjusted[3] <= adjusted[2]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
