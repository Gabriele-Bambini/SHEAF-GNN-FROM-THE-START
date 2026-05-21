"""Optuna adapter with injected objectives and CV splits."""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Callable, Mapping, Sequence

from sheaf_ts.eval.cross_validation import CVSplit

__all__ = ["SearchSpace", "TrialResult", "run_optuna_cv"]


@dataclass(frozen=True)
class SearchSpace:
    """Declarative Optuna search space."""

    floats: Mapping[str, tuple[float, float, bool]] = field(default_factory=dict)
    categoricals: Mapping[str, Sequence[object]] = field(default_factory=dict)
    integers: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def suggest(self, trial) -> dict[str, object]:
        params: dict[str, object] = {}
        for name, (low, high, log) in self.floats.items():
            params[name] = trial.suggest_float(name, low, high, log=log)
        for name, choices in self.categoricals.items():
            params[name] = trial.suggest_categorical(name, list(choices))
        for name, (low, high) in self.integers.items():
            params[name] = trial.suggest_int(name, low, high)
        return params


@dataclass(frozen=True)
class TrialResult:
    params: dict[str, object]
    fold_scores: tuple[float, ...]
    mean_score: float


def _validate_splits(splits: Sequence[CVSplit]) -> tuple[CVSplit, ...]:
    split_tuple = tuple(splits)
    if not split_tuple:
        raise ValueError("splits must be non-empty")
    for split in split_tuple:
        if not split.train_idx or not split.val_idx:
            raise ValueError("each split must contain train and validation indices")
        if set(split.train_idx) & set(split.val_idx):
            raise ValueError("train/validation indices overlap")
    return split_tuple


def run_optuna_cv(
    *,
    objective: Callable[[dict[str, object], CVSplit], float],
    search_space: SearchSpace,
    splits: Sequence[CVSplit],
    n_trials: int,
    direction: str = "maximize",
    study_name: str = "sheaf_ts_cv",
    storage: str | None = None,
    seed: int = 42,
    timeout: int | None = None,
) -> dict[str, object]:
    """Run Optuna over CV folds using a caller-provided objective.

    The objective receives `(params, split)` and must return a finite scalar.
    Optuna is imported inside this function only, so `sheaf_ts.hpo` remains
    importable in environments where Optuna is not installed.
    """
    split_tuple = _validate_splits(splits)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required to run HPO: pip install optuna") from exc

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, n_trials))
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=storage is not None,
    )
    trial_results: list[TrialResult] = []

    def optuna_objective(trial) -> float:
        params = search_space.suggest(trial)
        fold_scores = []
        for fold_idx, split in enumerate(split_tuple):
            score = float(objective(params, split))
            if score != score or score in {float("inf"), float("-inf")}:
                raise ValueError("objective returned a non-finite score")
            fold_scores.append(score)
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        mean_score = float(statistics.fmean(fold_scores))
        trial_results.append(
            TrialResult(
                params=dict(params),
                fold_scores=tuple(fold_scores),
                mean_score=mean_score,
            )
        )
        return mean_score

    study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    return {
        "study": study,
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "trial_results": trial_results,
    }
