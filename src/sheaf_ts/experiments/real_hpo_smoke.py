"""Tiny Optuna/CV smoke over the cached real-data sheaf predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheaf_ts.eval.cross_validation import CVSplit, temporal_blocked_splits
from sheaf_ts.experiments.real_data_smoke import (
    evaluate_real_transition_split,
    load_real_transition_cache,
)
from sheaf_ts.hpo.optuna_runner import SearchSpace, run_optuna_cv

__all__ = ["run_real_hpo_smoke"]


def _make_splits(num_bins: int, n_folds: int) -> tuple[CVSplit, ...]:
    num_transitions = num_bins - 1
    min_train = max(2, num_transitions - n_folds)
    splits = tuple(temporal_blocked_splits(n_bins=num_transitions, min_train_bins=min_train, horizon=1))
    return splits[:n_folds]


def run_real_hpo_smoke(
    *,
    temporal_path: str | Path,
    grn_path: str | Path,
    out_path: str | Path,
    run_id: str,
    n_trials: int = 10,
    n_folds: int = 2,
    seed: int = 42,
    allele: str | None = "WT",
    study_storage: str | Path | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    base_cache = load_real_transition_cache(
        temporal_path=temporal_path,
        grn_path=grn_path,
        allele=allele,
    )
    splits = _make_splits(base_cache["num_bins"], n_folds)
    if len(splits) != n_folds:
        raise ValueError("not enough temporal transitions for requested folds")

    search_space = SearchSpace(
        floats={
            "edge_weight_power": (0.5, 2.0, False),
            "ridge": (1e-6, 1e-2, True),
            "degree_regularizer": (1e-5, 1e-1, True),
        },
        categoricals={"use_temporal_edge_weights": (True, False)},
    )

    def objective(params: dict[str, object], split: CVSplit) -> float:
        cache = load_real_transition_cache(
            temporal_path=temporal_path,
            grn_path=grn_path,
            allele=allele,
            edge_weight_power=float(params["edge_weight_power"]),
        )
        result = evaluate_real_transition_split(
            cache,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            random_seed=seed,
            ridge=float(params["ridge"]),
            degree_regularizer=float(params["degree_regularizer"]),
            use_temporal_edge_weights=bool(params["use_temporal_edge_weights"]),
        )
        metric_name = "r_delta_perp_no_self"
        sheaf_score = float(result["model_metrics"]["sheaf_residual"][metric_name])
        control_score = max(
            float(metrics[metric_name])
            for metrics in result["baseline_metrics"].values()
        )
        return sheaf_score - control_score

    optuna_result = run_optuna_cv(
        objective=objective,
        search_space=search_space,
        splits=splits,
        n_trials=n_trials,
        direction="maximize",
        study_name=run_id,
        storage=f"sqlite:///{Path(study_storage)}" if study_storage else None,
        seed=seed,
        timeout=timeout,
    )
    trial_results = [
        {
            "params": trial.params,
            "fold_scores": list(trial.fold_scores),
            "mean_score": trial.mean_score,
        }
        for trial in optuna_result["trial_results"]
    ]
    manifest = {
        "run_id": run_id,
        "dataset": {
            "allele": base_cache["allele"],
            "num_bins": base_cache["num_bins"],
            "num_nodes": base_cache["num_nodes"],
            "num_edges": base_cache["num_edges"],
            "temporal_path": str(temporal_path),
            "grn_path": str(grn_path),
        },
        "cv": {
            "n_folds": n_folds,
            "splits": [
                {"fold": split.fold, "train_idx": list(split.train_idx), "val_idx": list(split.val_idx)}
                for split in splits
            ],
        },
        "best_params": optuna_result["best_params"],
        "best_value": optuna_result["best_value"],
        "objective": "r_delta_perp_no_self_sheaf_residual_minus_best_control",
        "study_storage": str(study_storage) if study_storage else None,
        "trial_results": trial_results,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return json.loads(json.dumps(manifest, sort_keys=True))
