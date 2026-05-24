"""
AnnData → tensor adapter for the Nature orchestrator.

Takes the heterogeneous AnnData outputs of the four registry datasets
(brca_wu2021, biokey, crc_pelka2021, metabric) and packs them into the
uniform tensor dict consumed by our compact training / evaluation pipeline:

    {
        "X_seq"        : (K, N, input_dim) float tensor,
        "edge_index"   : (2, E) long tensor,
        "edge_weight"  : (E,) float tensor (signed — negative = repressive),
        "gene_names"   : list[str] length N,
        "allele_label" : str (first obs-level allele column found, else "WT"),
        "grn_meta"     : dict with edge provenance (optional),
    }

Edge construction:
  - Default: construct_base_grn (TargetGeneReg + RegNetwork + STRING +
    Spearman weighting) plus DoRothEA A/B (ChIP-seq validated TF-target).
    Requires decoupler (pip install decoupler).
    Raises RuntimeError if zero biological edges found — never silently
    degrades to kNN.
  - Optional (use_scprint_grn=True): scPRINT attention-derived GRN —
    biologically grounded, causality-informed, parquet-cached.
    Falls back to biological priors (not kNN) if scPRINT unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── AnnData normalization ───────────────────────────────────────────────────

def _as_dense(X):
    """Return a dense float32 ndarray (cells × genes)."""
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _pick_allele_column(adata) -> Optional[str]:
    for col in ("tp53_allele_canonical", "allele", "tp53_allele", "allele_tag"):
        if col in adata.obs.columns:
            return col
    return None


# ─── Temporal binning (K pseudotime groups) ──────────────────────────────────

def _pseudotime_bins(X: np.ndarray, K: int, rng, method: str = "mean_expr",
                     adata=None) -> list:
    """Return K index-lists over rows of X.

    Methods:
      - "mean_expr": rank cells by mean expression (original, unreliable for
        homogeneous datasets — adjacent bins become nearly identical which
        makes the forecast task trivially solved by identity baselines).
      - "dpt": diffusion pseudotime via scanpy. Biologically grounded, uses
        graph-diffusion from the cell with lowest total counts as root.
        Falls back to mean_expr if scanpy fails or adata is None.
      - "shuffled": random permutation — control baseline. Forecast models
        trained/evaluated on shuffled bins should LOSE signal. If they don't,
        the pseudotime axis isn't driving predictive power.
    """
    n = X.shape[0]
    if K <= 0 or n < K:
        return [np.arange(n)]

    if method == "shuffled":
        order = rng.permutation(n)
    elif method == "dpt" and adata is not None:
        try:
            import scanpy as sc
            import anndata as ad
            a = adata.copy() if hasattr(adata, "copy") else ad.AnnData(X.copy())
            sc.pp.pca(a, n_comps=min(30, X.shape[1]-1, n-1), zero_center=False)
            sc.pp.neighbors(a, n_neighbors=min(15, n-1), use_rep="X_pca")
            sc.tl.diffmap(a, n_comps=min(15, n-1))
            # Root = cell with lowest total expression (proxy for earliest)
            root = int(np.argmin(X.sum(axis=1)))
            a.uns["iroot"] = root
            sc.tl.dpt(a)
            pt = np.asarray(a.obs["dpt_pseudotime"].values, dtype=np.float64)
            # Replace any NaN/inf with median
            if not np.all(np.isfinite(pt)):
                med = np.nanmedian(pt[np.isfinite(pt)]) if np.any(np.isfinite(pt)) else 0.0
                pt = np.where(np.isfinite(pt), pt, med)
            order = np.argsort(pt)
            logger.info("[adapter] DPT pseudotime: min=%.3f max=%.3f", pt.min(), pt.max())
        except Exception as e:
            logger.warning("[adapter] DPT failed (%s) — falling back to mean_expr", e)
            order = np.argsort(X.mean(axis=1))
    else:
        order = np.argsort(X.mean(axis=1))

    bin_size = n // K
    bins = [order[i * bin_size: (i + 1) * bin_size] for i in range(K - 1)]
    bins.append(order[(K - 1) * bin_size:])
    return bins


# ─── Gene selection with TF protection ──────────────────────────────────────

def select_top_genes_with_tf_protection(
    X: np.ndarray,
    gene_names: list,
    n_genes: int = 500,
) -> list:
    """
    Select top n_genes by variance, but ALWAYS include BRCA key TFs.

    Without this, low-expressed TFs (FOXA1, ESR1, etc.) are excluded by
    standard HVG selection. The sheaf cannot model TP53 regulation if the
    regulator nodes are absent.

    Strategy: union(top (n_genes - n_protected) HVG, BRCA_KEY_TFS ∩ gene_names)
    """
    from tsgnn.data.grn_construction import BRCA_KEY_TFS

    gene_set = set(gene_names)
    protected = [g for g in BRCA_KEY_TFS if g in gene_set]

    n_hvg = max(1, n_genes - len(protected))
    variances = X.var(axis=0)
    protected_idx = {gene_names.index(g) for g in protected}
    masked_var = variances.copy()
    for idx in protected_idx:
        masked_var[idx] = -1.0  # exclude from top-k

    top_idx = np.argpartition(-masked_var, n_hvg)[:n_hvg]
    hvg_genes = [gene_names[i] for i in top_idx]

    selected = list(dict.fromkeys(protected + hvg_genes))[:n_genes]
    return selected


# ─── GRN edge construction (construct_base_grn + DoRothEA, no kNN) ──────────

def _build_grn(
    X_mean: np.ndarray,
    gene_list: list,
) -> tuple:
    """
    Build GRN from all available biological priors via construct_base_grn,
    plus DoRothEA A/B edges (ChIP-seq validated TF-target).

    Convergent path: uses the same construct_base_grn used by run_pipeline.py,
    plus DoRothEA which construct_base_grn doesn't include natively.

    Returns:
        (edge_index, edge_weight, grn_meta) where:
        - edge_index: (2, E) long tensor
        - edge_weight: (E,) float tensor (signed — negative = repressive)
        - grn_meta: dict with edge provenance and node info
    Raises RuntimeError if zero biological edges found.
    """
    from tsgnn.data.grn_construction import (
        construct_base_grn,
        load_dorothea_prior_edges,
        load_targetgenereg,
        load_regnetwork,
        load_string_ppi,
    )

    # ── Load available priors ────────────────────────────────────────────────
    tgr_df = load_targetgenereg()
    try:
        regnet_df = load_regnetwork()
    except Exception:
        regnet_df = None

    try:
        from pathlib import Path as _P
        import os
        _data_dir = _P(os.environ.get("TSGNN_DATA_DIR", "data/external"))
        _str_dir = _data_dir / "string"
        _str_cands = sorted(_str_dir.glob("*protein.links*")) if _str_dir.exists() else []
        string_df = load_string_ppi(_str_cands[0]) if _str_cands else None
    except Exception:
        string_df = None

    # ── construct_base_grn: TargetGeneReg + RegNetwork + STRING + Spearman ──
    edge_index, edge_weight, grn_meta = construct_base_grn(
        gene_list=gene_list,
        regnetwork_edges=regnet_df,
        targetgenereg=tgr_df if len(tgr_df) > 0 else None,
        string_ppi=string_df,
        expr_matrix=X_mean,  # enables Spearman weighting
    )

    n_base = edge_index.shape[1]
    logger.info("[adapter] construct_base_grn: %d edges", n_base)

    # ── DoRothEA A/B: add ChIP-seq validated edges not already in GRN ──────
    dorothea_edges = load_dorothea_prior_edges(gene_list, levels=["A", "B"])
    if dorothea_edges:
        existing = set()
        for i in range(edge_index.shape[1]):
            existing.add((int(edge_index[0, i]), int(edge_index[1, i])))

        new_srcs, new_tgts, new_weights = [], [], []
        for (s, t), w in dorothea_edges.items():
            if (s, t) not in existing:
                new_srcs.append(s)
                new_tgts.append(t)
                new_weights.append(w)
                existing.add((s, t))

        if new_srcs:
            new_ei = torch.tensor([new_srcs, new_tgts], dtype=torch.long)
            new_ew = torch.tensor(new_weights, dtype=torch.float32)
            edge_index = torch.cat([edge_index, new_ei], dim=1)
            edge_weight = torch.cat([edge_weight, new_ew])
            logger.info("[adapter] DoRothEA A/B: +%d new edges (total %d)",
                        len(new_srcs), edge_index.shape[1])

        # Update metadata
        grn_meta["n_edges"] = edge_index.shape[1]
        for (s, t), w in dorothea_edges.items():
            grn_meta.setdefault("edge_sources", {})[(s, t)] = "DoRothEA"

    total = edge_index.shape[1]
    if total == 0:
        raise RuntimeError(
            f"No biological GRN edges found for gene set of size {len(gene_list)}. "
            "Check that decoupler is installed (pip install decoupler) and that "
            "DoRothEA/TargetGeneReg data files are accessible. "
            "Refusing to fall back to kNN — fix the data pipeline."
        )

    logger.info("[adapter] Total biological GRN: %d edges", total)
    return edge_index, edge_weight, grn_meta


# Backward-compatible alias
def _build_grn_edges(X_mean: np.ndarray, gene_list: list, k_neighbors: int = 8) -> torch.Tensor:
    """DEPRECATED: use _build_grn() which returns (edge_index, edge_weight, meta)."""
    edge_index, _ew, _meta = _build_grn(X_mean, gene_list)
    return edge_index


# ─── kNN correlation edges ───────────────────────────────────────────────────

def _correlation_knn_edges(X: np.ndarray, k: int = 8) -> torch.Tensor:
    """Return (2, E) edge_index from gene×gene abs-Pearson kNN (directed, no self).

    Uses a memory-friendly path for large gene counts: z-scores × matmul.
    """
    N = X.shape[1]
    Xg = X.T                                     # (N, cells)
    Xg = Xg - Xg.mean(axis=1, keepdims=True)
    std = Xg.std(axis=1, keepdims=True) + 1e-8
    Xg = Xg / std
    C = np.abs((Xg @ Xg.T) / Xg.shape[1])        # (N, N)
    np.fill_diagonal(C, 0.0)
    k = max(1, min(k, N - 1))
    top_k = np.argpartition(-C, k, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = top_k.flatten()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ─── Public API ──────────────────────────────────────────────────────────────

def adata_to_tsgnn_tensors(
    adata,
    K: int = 10,
    n_genes: int = 500,
    k_neighbors: int = 8,
    input_dim: int = 1,
    seed: int = 0,
    use_scprint_grn: bool = False,
    scprint_cache_dir: Optional[Path] = None,
    scprint_dataset_tag: str = "dataset",
    scprint_ckpt: str = "jkobject/scPRINT",
    ref_mu: Optional[np.ndarray] = None,
    ref_sd: Optional[np.ndarray] = None,
    ref_genes: Optional[List[str]] = None,
    pseudotime_method: str = "mean_expr",
) -> Dict:
    """Build the tensor bundle consumed by the Nature orchestrator.

    Args:
        adata: AnnData (cells|samples × genes).
        K: pseudotime bins.
        n_genes: top-variance genes to keep.
        k_neighbors: unused (kept for API compat). Biological GRN ignores kNN.
        input_dim: feature dimension per gene (1 = mean expression).
        seed: reproducibility knob for any stochastic fallback.
        use_scprint_grn: If True, build edges from scPRINT GRN attention
            weights instead of Pearson kNN. Falls back to kNN if scPRINT
            is not installed or inference fails.
        scprint_cache_dir: Directory for scPRINT GRN parquet cache.
            Defaults to data/processed/ relative to adata location.
        scprint_dataset_tag: Short identifier used in cache filename.
        scprint_ckpt: HuggingFace checkpoint for scPRINT model.
        ref_mu: Optional reference mean for Z-scoring (fixes transductive leakage).
        ref_sd: Optional reference std for Z-scoring (fixes transductive leakage).
    """
    rng = np.random.default_rng(seed)
    X = _as_dense(adata.X)                       # (cells, genes)
    gene_names = list(adata.var_names.astype(str))

    # Normalize: library-size → log1p → gene-wise z-score
    lib = X.sum(axis=1, keepdims=True).clip(min=1)
    X = np.log1p(X / lib * 1e4)                 # log-normalise (CPM-like)
    
    if ref_mu is not None and ref_sd is not None:
        # Inductive normalization (Noble/Nature standard)
        # Re-index genes if ref_genes provided (cross-dataset case)
        if ref_genes is not None:
            ref_idx = {g: i for i, g in enumerate(ref_genes)}
            keep_idx = [i for i, g in enumerate(gene_names) if g in ref_idx]
            gene_mask = [ref_idx.get(gene_names[i], -1) for i in keep_idx]
            if not keep_idx:
                logger.warning("[adapter] No gene overlap with reference — skipping inductive normalization.")
            else:
                X = X[:, keep_idx]
                gene_names = [gene_names[i] for i in keep_idx]
                ref_mu_aligned = ref_mu[:, gene_mask]
                ref_sd_aligned = ref_sd[:, gene_mask]
                X = (X - ref_mu_aligned) / (ref_sd_aligned + 1e-8)
                logger.info(f"[adapter] Applied inductive normalization using {len(keep_idx)}/{len(ref_genes)} reference genes.")
        else:
            X = (X - ref_mu) / (ref_sd + 1e-8)
            logger.info(f"[adapter] Applied inductive normalization using reference stats.")
    else:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mu) / sd                           # z-score per gene
        # Return stats for future inductive use
        ref_mu, ref_sd = mu, sd

    # Feature selection: top-variance (after normalisation)
    var = X.var(axis=0)
    # Proving myself wrong: Top-variance selection is standard, BUT for TP53
    # allele-conditioning to be biologically meaningful, TP53 MUST be in the node set.
    # We force TP53 to be included even if it's low variance.
    tp53_idx = next((i for i, g in enumerate(gene_names) if g == "TP53"), None)
    
    keep = np.argsort(var)[::-1]
    keep_list = []
    if tp53_idx is not None:
        keep_list.append(tp53_idx)
    
    for idx in keep:
        if len(keep_list) >= n_genes:
            break
        if idx not in keep_list:
            keep_list.append(idx)
            
    keep = np.array(keep_list)
    X = X[:, keep]
    gene_names = [gene_names[i] for i in keep]
    
    # Store used stats for downstream inference alignment.
    # If ref_mu was computed locally (full-gene, else-branch), slice to the
    # selected genes so shape matches gene_names (n_genes, not full vocab).
    if ref_mu is not None and ref_mu.shape[1] != len(gene_names):
        stats = {"mu": ref_mu[:, keep], "sd": ref_sd[:, keep], "genes": gene_names}
    else:
        stats = {"mu": ref_mu, "sd": ref_sd, "genes": gene_names}

    # Temporal binning (pseudotime)
    bins = _pseudotime_bins(X, K, rng, method=pseudotime_method, adata=adata)
    X_bins = np.stack([X[idx].mean(axis=0) if len(idx) else np.zeros(X.shape[1], np.float32)
                       for idx in bins])        # (K, N)
    X_seq = torch.tensor(X_bins, dtype=torch.float32).unsqueeze(-1)
    if input_dim > 1:
        X_seq = X_seq.expand(-1, -1, input_dim).clone()

    # Edge construction: scPRINT GRN (optional) or Pearson kNN (default)
    if use_scprint_grn:
        from tsgnn.data.scprint_grn import get_scprint_grn_edges
        # Provide AnnData filtered to selected genes for inference efficiency
        import anndata
        adata_sub = adata[:, [adata.var_names.get_loc(g)
                               if hasattr(adata.var_names, 'get_loc')
                               else list(adata.var_names).index(g)
                               for g in gene_names
                               if g in adata.var_names]].copy()
        edge_index = get_scprint_grn_edges(
            adata_sub,
            gene_names=gene_names,
            k=k_neighbors,
            cache_dir=scprint_cache_dir,
            dataset_tag=scprint_dataset_tag,
            ckpt=scprint_ckpt,
        )
        logger.info(f"[adapter] scPRINT GRN: {edge_index.shape[1]} edges")
        # scPRINT returns only edge_index; uniform weights, no provenance
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        grn_meta = {"n_edges": edge_index.shape[1], "edge_sources": {}}
    else:
        edge_index, edge_weight, grn_meta = _build_grn(X, gene_names)

    allele_col = _pick_allele_column(adata)
    allele_label = "WT"
    if allele_col is not None:
        vc = adata.obs[allele_col].astype(str).value_counts()
        if len(vc):
            allele_label = str(vc.index[0])

    bundle = {
        "X_seq": X_seq,
        "edge_index": edge_index,
        "gene_names": gene_names,
        "allele_label": allele_label,
        "stats": stats,
    }
    if edge_weight is not None:
        bundle["edge_weight"] = edge_weight
    if grn_meta is not None:
        bundle["grn_meta"] = grn_meta
    return bundle


def align_to_gene_set(bundle: Dict, reference_genes: Iterable[str]) -> Dict:
    """Project a bundle's X_seq onto `reference_genes` (missing → 0).

    DEPRECATED: Use intersect_gene_sets() instead. Zero-filling creates
    spurious correlations and corrupts edge construction.
    """
    ref = list(reference_genes)
    ref_to_idx = {g: i for i, g in enumerate(ref)}
    src_to_idx = {g: i for i, g in enumerate(bundle["gene_names"])}
    common = [g for g in ref if g in src_to_idx]
    if not common:
        logger.warning("align_to_gene_set: zero gene overlap — returning zeros.")
    K, _, D = bundle["X_seq"].shape
    new_X = torch.zeros(K, len(ref), D)
    for g in common:
        new_X[:, ref_to_idx[g], :] = bundle["X_seq"][:, src_to_idx[g], :]
    X_np = new_X.mean(dim=-1).numpy()
    edge_index, edge_weight, grn_meta = _build_grn(X_np, ref)
    result = {
        "X_seq": new_X,
        "edge_index": edge_index,
        "gene_names": ref,
        "allele_label": bundle.get("allele_label", "WT"),
    }
    if edge_weight is not None:
        result["edge_weight"] = edge_weight
    return result


def intersect_gene_sets(
    bundle_source: Dict,
    bundle_target: Dict,
    min_genes: int = 200,
    k_neighbors: int = 8,
) -> tuple:
    """
    Intersect gene sets between source and target bundles.

    Unlike align_to_gene_set (which zero-fills missing genes and corrupts
    correlations), this function subsets BOTH datasets to the exact
    intersection of their gene sets.

    Args:
        bundle_source: Source dataset bundle (e.g., BRCA).
        bundle_target: Target dataset bundle (e.g., CRC).
        min_genes: Minimum intersection size. Fails if overlap < min_genes.
        k_neighbors: kNN parameter for edge reconstruction.

    Returns:
        (source_intersected, target_intersected, common_genes)
        Both bundles have identical gene ordering, freshly built edges,
        and only contain genes present in both datasets.

    Raises:
        ValueError: If gene overlap < min_genes.
    """
    src_genes = bundle_source["gene_names"]
    tgt_genes = bundle_target["gene_names"]

    src_set = set(src_genes)
    tgt_set = set(tgt_genes)
    common = sorted(src_set & tgt_set)

    if len(common) < min_genes:
        raise ValueError(
            f"Gene intersection too small: {len(common)} < {min_genes}. "
            f"Source has {len(src_genes)}, target has {len(tgt_genes)}."
        )

    logger.info(
        f"intersect_gene_sets: {len(src_genes)} ∩ {len(tgt_genes)} = "
        f"{len(common)} common genes"
    )

    def _subset_bundle(bundle: Dict, genes: List[str]) -> Dict:
        """Subset a bundle to a specific gene ordering."""
        name_to_idx = {g: i for i, g in enumerate(bundle["gene_names"])}
        indices = [name_to_idx[g] for g in genes]
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        X_seq = bundle["X_seq"][:, idx_tensor, :]  # (K, N_common, D)

        # Rebuild biological GRN on the intersected gene set
        X_np = X_seq.mean(dim=-1).numpy()  # (K, N_common)
        try:
            edge_index, edge_weight, grn_meta = _build_grn(X_np, genes)
        except RuntimeError:
            # Fallback for synthetic/test data with no biological priors
            logger.warning("[intersect] No biological priors for gene set — "
                           "falling back to correlation kNN edges")
            edge_index = _correlation_knn_edges(X_np, k=min(k_neighbors, len(genes) - 1))
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        result = {
            "X_seq": X_seq,
            "edge_index": edge_index,
            "gene_names": genes,
            "allele_label": bundle.get("allele_label", "WT"),
        }
        if edge_weight is not None:
            result["edge_weight"] = edge_weight
        return result

    src_out = _subset_bundle(bundle_source, common)
    tgt_out = _subset_bundle(bundle_target, common)

    return src_out, tgt_out, common
