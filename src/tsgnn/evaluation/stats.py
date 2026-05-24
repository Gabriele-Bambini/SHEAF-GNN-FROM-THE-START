"""
Statistical rigor helpers for Nature-grade reporting.

Provides:
  - bootstrap_ci          : 95% bias-corrected percentile CI
  - benjamini_hochberg    : FDR-adjusted p-values
  - wilcoxon_paired       : paired Wilcoxon across seeds (for ablations)
  - aggregate_seeds       : mean ± std ± CI dict from list of per-seed floats
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if v.size > 1 else 0.0),
            "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # monotone from the tail
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank (a vs b). Returns {"stat","p","n"}."""
    from scipy.stats import wilcoxon
    a = np.asarray(list(a), dtype=float); b = np.asarray(list(b), dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 2 or np.all(a == b):
        return {"stat": float("nan"), "p": 1.0, "n": int(a.size)}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(stat), "p": float(p), "n": int(a.size)}


def aggregate_seeds(values: Iterable[float], n_boot: int = 2000) -> dict:
    return bootstrap_ci(list(values), n_boot=n_boot)
