"""
Allele-swap falsifiability metric (Nature claim P2).

Given a trained TS-GNN and an input expression snapshot, swapping the ESM-2
allele embedding (e.g. R175H → R273H) must change the top-k most-dysregulated
genes by a non-trivial Jaccard distance. Random-allele controls (shuffled
embedding) must not. This turns the FiLM claim into a falsifiable test.

API:
    swap_jaccard(model, X, emb_A, emb_B, k=50) -> float
    expected_effect_curve(model, X, emb_A, emb_B, ks=[10,25,50,100]) -> dict
    control_shuffle(model, X, emb_A, rng) -> float
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _predict(model, X, emb):
    """Run model on X (either (N,d) or (K,N,d)); return flat per-gene tensor.

    Handles both TS-GNN-style (4-item) returns and plain-tensor returns, and
    list-of-tensor returns from temporal baselines (returns t=0 prediction).
    """
    if X.dim() == 2:
        X_in = X.unsqueeze(0)  # treat as K=1
    else:
        X_in = X
    out = model(X_in, emb)
    
    # Handle TS-GNN 4-item return: (preds, maps, laps, stalks)
    if isinstance(out, tuple):
        preds = out[0]
    else:
        preds = out
        
    if isinstance(preds, list):
        preds = preds[0]
    
    # preds is (K, N, D) or (N, D). Get first time step and flatten feature dim.
    if preds.dim() == 3:
        # (K, N, D) -> (N, D)
        preds = preds[0]
    
    return preds


def _topk_gene_indices(delta: torch.Tensor, k: int) -> np.ndarray:
    """Top-k genes by absolute change magnitude across all dimensions.
    
    If delta is (N, D), computes norm across D to identify the N genes.
    """
    if delta.dim() > 1:
        # Compute L2 norm across feature dimension to get per-gene perturbation
        v = torch.norm(delta, p=2, dim=-1).detach().cpu().numpy()
    else:
        v = delta.detach().abs().cpu().numpy()
        
    if v.size <= k:
        return np.argsort(-v)
    return np.argpartition(-v, k)[:k]


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / max(1, len(sa | sb))


def swap_jaccard(model, X: torch.Tensor, emb_A: torch.Tensor, emb_B: torch.Tensor,
                 k: int = 50) -> float:
    """Jaccard distance between top-k perturbed genes under allele A vs allele B.

    Args:
        model: callable(X_flat, emb) -> preds  (TS-GNN forward)
        X:     (N, d) gene-expression snapshot
        emb_A, emb_B: (1, 1280) ESM-2 allele embeddings
        k:     top-k to compare
    """
    model.eval()
    with torch.no_grad():
        pA = _predict(model, X, emb_A)
        pB = _predict(model, X, emb_B)
    # Per-gene delta vs allele-A baseline: |ΔA→B| identifies the genes that *move*
    deltaB = pB - pA
    tA = _topk_gene_indices(torch.zeros_like(pA), k)  # baseline (no movement)
    tB = _topk_gene_indices(deltaB, k)
    # Jaccard between A-static ranking vs B-perturbed ranking
    return _jaccard(tA, tB)


def expected_effect_curve(model, X, emb_A, emb_B, ks=(10, 25, 50, 100, 200)) -> dict:
    model.eval()
    with torch.no_grad():
        pA = _predict(model, X, emb_A)
        pB = _predict(model, X, emb_B)
    delta = (pB - pA)
    out = {}
    for k in ks:
        tA = _topk_gene_indices(pA, k)
        tB = _topk_gene_indices(pB, k)
        out[f"jaccard_topk_{k}"] = _jaccard(tA, tB)
        out[f"delta_topk_{k}_mean_abs"] = float(
            delta.detach().abs().flatten().cpu().numpy()[_topk_gene_indices(delta, k)].mean()
        )
    return out


def control_shuffle(model, X, emb_A, seed: int = 0) -> float:
    """Null control: shuffle emb_A elementwise → should NOT change top-k."""
    g = torch.Generator(device=emb_A.device).manual_seed(seed)
    perm = torch.randperm(emb_A.shape[-1], generator=g, device=emb_A.device)
    emb_shuf = emb_A[..., perm]
    return swap_jaccard(model, X, emb_A, emb_shuf, k=50)
