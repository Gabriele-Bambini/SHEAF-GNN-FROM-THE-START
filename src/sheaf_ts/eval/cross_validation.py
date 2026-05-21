"""Deterministic cross-validation splitters with leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Iterator, Sequence

__all__ = ["CVSplit", "repeated_group_kfold", "temporal_blocked_splits"]


@dataclass(frozen=True)
class CVSplit:
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    fold: int
    repeat: int = 0
    kind: str = "group_kfold"


def _group_to_indices(groups: Sequence[object]) -> dict[object, list[int]]:
    mapping: dict[object, list[int]] = {}
    for idx, group in enumerate(groups):
        mapping.setdefault(group, []).append(idx)
    return mapping


def repeated_group_kfold(
    groups: Sequence[object],
    *,
    n_splits: int,
    n_repeats: int = 1,
    seed: int = 0,
) -> Iterator[CVSplit]:
    """Yield deterministic group-disjoint repeated K-fold splits."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    grouped = _group_to_indices(groups)
    unique_groups = list(grouped)
    if len(unique_groups) < n_splits:
        raise ValueError("number of unique groups must be >= n_splits")

    for repeat in range(n_repeats):
        rng = random.Random(seed + repeat)
        shuffled = unique_groups[:]
        rng.shuffle(shuffled)
        folds = [shuffled[i::n_splits] for i in range(n_splits)]

        for fold_id, val_groups in enumerate(folds):
            val_group_set = set(val_groups)
            train_idx = []
            val_idx = []
            for group, indices in grouped.items():
                if group in val_group_set:
                    val_idx.extend(indices)
                else:
                    train_idx.extend(indices)
            if not train_idx or not val_idx:
                raise ValueError("split produced empty train or validation set")
            yield CVSplit(
                train_idx=tuple(sorted(train_idx)),
                val_idx=tuple(sorted(val_idx)),
                fold=fold_id,
                repeat=repeat,
                kind="group_kfold",
            )


def temporal_blocked_splits(
    *,
    n_bins: int,
    min_train_bins: int,
    horizon: int = 1,
) -> Iterator[CVSplit]:
    """Yield expanding-window temporal splits.

    Validation bins always occur after all training bins.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if min_train_bins < 1:
        raise ValueError("min_train_bins must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if min_train_bins + horizon > n_bins:
        raise ValueError("not enough bins for requested split")

    fold = 0
    start = min_train_bins
    while start + horizon <= n_bins:
        train_idx = tuple(range(start))
        val_idx = tuple(range(start, start + horizon))
        yield CVSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            fold=fold,
            repeat=0,
            kind="temporal_blocked",
        )
        fold += 1
        start += horizon
