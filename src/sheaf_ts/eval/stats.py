"""Small statistical helpers for multi-seed and biological reports."""

from __future__ import annotations

import random
from typing import Sequence

__all__ = ["benjamini_hochberg", "bootstrap_mean_ci"]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Deterministic bootstrap confidence interval for the mean."""
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("values must be non-empty")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be inside (0, 1)")
    if any(value != value or value in {float("inf"), float("-inf")} for value in sample):
        raise ValueError("values must be finite")

    rng = random.Random(seed)
    means = []
    n = len(sample)
    for _ in range(n_bootstrap):
        draw = [sample[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    alpha = 1.0 - confidence
    low_idx = max(0, int((alpha / 2.0) * n_bootstrap))
    high_idx = min(n_bootstrap - 1, int((1.0 - alpha / 2.0) * n_bootstrap))
    return {
        "mean": sum(sample) / n,
        "low": means[low_idx],
        "high": means[high_idx],
        "confidence": confidence,
        "n": float(n),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-adjusted p-values in the original order."""
    values = [float(value) for value in p_values]
    if any(value < 0.0 or value > 1.0 or value != value for value in values):
        raise ValueError("p-values must be finite values in [0, 1]")
    m = len(values)
    if m == 0:
        return []

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(running_min, p_value * m / rank)
        running_min = adjusted
        adjusted_sorted[rank - 1] = adjusted

    adjusted = [0.0] * m
    for sorted_pos, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = min(1.0, adjusted_sorted[sorted_pos])
    return adjusted
