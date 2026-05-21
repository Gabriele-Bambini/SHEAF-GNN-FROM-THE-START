"""
METABRIC bulk-enrichment metric (Nature claim P3).

For each TP53 hotspot allele, test whether the gene set predicted downstream
by TS-GNN (given the allele's ESM-2 embedding) is differentially expressed in
METABRIC tumors carrying that allele vs WT. Reports a BH-corrected q-value
per allele and a consolidated "enrichment passes P3" flag.

Pipeline (per allele a):
  1. Get TS-GNN prediction vector for allele a; take top-N up/down genes → S_a.
  2. In METABRIC, compute per-gene Welch-t between cohort(a) and cohort(WT).
  3. Restrict to genes in S_a; compute rank-sum enrichment vs genome background.
  4. BH-correct the per-allele enrichment p-values.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg


def _welch_t_per_gene(X_a: np.ndarray, X_wt: np.ndarray) -> tuple:
    """Return (t-stat, p-val) arrays of length G. Safe on zero-var genes."""
    from scipy.stats import ttest_ind
    t, p = ttest_ind(X_a, X_wt, axis=0, equal_var=False, nan_policy="omit")
    t = np.asarray(t, dtype=float); p = np.asarray(p, dtype=float)
    t = np.where(np.isfinite(t), t, 0.0)
    p = np.where(np.isfinite(p), p, 1.0)
    return t, p


def _rank_sum_enrichment(scores: np.ndarray, hit_mask: np.ndarray) -> float:
    """One-sided Mann-Whitney: are 'hit' genes ranked higher in |score| than others?"""
    from scipy.stats import mannwhitneyu
    if hit_mask.sum() < 3 or (~hit_mask).sum() < 3:
        return 1.0
    try:
        _, p = mannwhitneyu(np.abs(scores[hit_mask]), np.abs(scores[~hit_mask]),
                             alternative="greater")
    except ValueError:
        return 1.0
    return float(p)


def enrichment_per_allele(
    metabric_adata,
    predicted_sets: Dict[str, Iterable[str]],
    wt_label: str = "WT",
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Test each allele's predicted gene set for DE-enrichment in METABRIC.

    Args:
        metabric_adata: AnnData with obs["tp53_allele_canonical"]; var_names = Hugo.
        predicted_sets: {allele → iterable of Hugo symbols predicted as downstream}.
        wt_label: key in the obs column treated as reference cohort.
        fdr: BH FDR for multi-allele correction.

    Returns:
        DataFrame with cols: allele, n_hits_in_var, raw_p, q_bh, passes_p3
    """
    var_names = metabric_adata.var_names.astype(str)
    var_to_idx = {g: i for i, g in enumerate(var_names)}
    wt_mask = metabric_adata.obs["tp53_allele_canonical"].astype(str).values == wt_label
    X_wt = metabric_adata.X[wt_mask]

    rows = []
    raw_ps: list = []
    for allele, genes in predicted_sets.items():
        a_mask = metabric_adata.obs["tp53_allele_canonical"].astype(str).values == allele
        if a_mask.sum() < 3:
            rows.append({"allele": allele, "n_hits_in_var": 0, "raw_p": 1.0,
                         "n_samples_allele": int(a_mask.sum())})
            raw_ps.append(1.0)
            continue
        X_a = metabric_adata.X[a_mask]
        t, _ = _welch_t_per_gene(X_a, X_wt)
        hit_idx = [var_to_idx[g] for g in genes if g in var_to_idx]
        hit_mask = np.zeros(t.shape[0], dtype=bool); hit_mask[hit_idx] = True
        p = _rank_sum_enrichment(t, hit_mask)
        raw_ps.append(p)
        rows.append({"allele": allele, "n_hits_in_var": int(hit_mask.sum()),
                     "raw_p": p, "n_samples_allele": int(a_mask.sum())})
    df = pd.DataFrame(rows)
    df["q_bh"] = benjamini_hochberg(df["raw_p"].values)
    df["passes_p3"] = df["q_bh"] < fdr
    return df.sort_values("q_bh").reset_index(drop=True)
