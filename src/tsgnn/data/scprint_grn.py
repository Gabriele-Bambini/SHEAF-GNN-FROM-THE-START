"""
scPRINT-based GRN inference for TS-GNN edge construction.

Replaces (or augments) the Pearson-kNN graph with a biologically grounded
gene regulatory network inferred from scPRINT transformer attention weights.

scPRINT (Drouhard et al. 2024, https://github.com/jkobject/scPRINT) is a
single-cell foundation model whose cross-attention heads encode gene-gene
regulatory interactions. We extract the top-k attention-derived edges per
gene and cache them as parquet to avoid re-running inference on every seed.

Install:
    pip install scprint
    # model checkpoint auto-downloaded from HuggingFace on first run

Usage (internal — called from nature_adapter.py):
    edge_index = get_scprint_grn_edges(adata, gene_names, k=8, cache_dir=...)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default HuggingFace checkpoint for scPRINT
_SCPRINT_CKPT = "jkobject/scPRINT"

# Max cells to pass to scPRINT inference (memory cap)
_MAX_CELLS_INFERENCE = 2000


def _scprint_available() -> bool:
    try:
        import scprint  # noqa: F401
        return True
    except ImportError:
        return False


def _grn_cache_path(cache_dir: Path, dataset_tag: str, n_genes: int) -> Path:
    return cache_dir / f"_scprint_grn_{dataset_tag}_n{n_genes}.parquet"


def _run_scprint_inference(adata, gene_names: List[str], max_cells: int, ckpt: str):
    """Run scPRINT GRN inference → DataFrame with columns (source, target, weight)."""
    import scprint
    from scprint.tasks.grn_inference import GRNfer

    # Subsample cells to bound RAM / compute time
    if adata.n_obs > max_cells:
        import numpy as _np
        idx = _np.random.default_rng(0).choice(adata.n_obs, size=max_cells, replace=False)
        adata_sub = adata[idx].copy()
    else:
        adata_sub = adata

    # Ensure var_names match scPRINT expectations (ENSEMBL or Hugo)
    logger.info(f"[scprint_grn] Loading checkpoint: {ckpt}")
    try:
        from scprint import scPrint
        model = scPrint.load_from_checkpoint(ckpt)
    except Exception as e:
        raise RuntimeError(f"scPRINT model load failed: {e}") from e

    logger.info(
        f"[scprint_grn] Running GRN inference on {adata_sub.n_obs} cells × "
        f"{adata_sub.n_vars} genes ..."
    )
    grner = GRNfer(
        model,
        adata_sub,
        num_workers=0,
        max_cells=max_cells,
        doplot=False,
    )
    grner()

    # GRNfer.GRNs is a list of DataFrames (one per attention head / layer);
    # aggregate by mean weight across heads.
    import pandas as pd
    frames = grner.GRNs if hasattr(grner, "GRNs") else []
    if not frames:
        raise RuntimeError("scPRINT GRNfer returned no GRNs.")

    # Each frame typically has columns: ['TF', 'gene', 'score'] or ['source', 'target', 'weight']
    # Normalise column names defensively.
    normed = []
    for df in frames:
        df = df.copy()
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("tf", "source", "gene1", "regulator"):
                col_map[c] = "source"
            elif cl in ("gene", "target", "gene2"):
                col_map[c] = "target"
            elif cl in ("score", "weight", "importance", "value"):
                col_map[c] = "weight"
        df = df.rename(columns=col_map)
        if {"source", "target", "weight"}.issubset(df.columns):
            normed.append(df[["source", "target", "weight"]])

    if not normed:
        raise RuntimeError(
            f"Cannot parse scPRINT GRN columns. Got: {[list(f.columns) for f in frames[:3]]}"
        )

    import pandas as pd
    combined = pd.concat(normed).groupby(["source", "target"], as_index=False)["weight"].mean()
    return combined


def get_scprint_grn_edges(
    adata,
    gene_names: List[str],
    k: int = 8,
    cache_dir: Optional[Path] = None,
    dataset_tag: str = "dataset",
    ckpt: str = _SCPRINT_CKPT,
    max_cells: int = _MAX_CELLS_INFERENCE,
) -> torch.Tensor:
    """Return (2, E) edge_index from scPRINT GRN, restricted to gene_names.

    Pipeline:
    1. Check parquet cache → return if hit.
    2. Run scPRINT GRNfer inference on AnnData (subsampled to max_cells).
    3. Filter edges: both source AND target must be in gene_names.
    4. Keep top-k outgoing edges per gene by weight (same convention as Pearson kNN).
    5. Cache result as parquet.
    6. Return (2, E) LongTensor.

    Falls back to Pearson kNN with a warning if scPRINT is unavailable or fails.

    Args:
        adata:        AnnData (cells × genes). var_names must cover gene_names.
        gene_names:   Ordered list of N gene names (top-variance subset).
        k:            Max outgoing edges per gene.
        cache_dir:    Directory for parquet cache (data/processed/ by default).
        dataset_tag:  Short string identifying the dataset (used in cache filename).
        ckpt:         HuggingFace checkpoint identifier for scPRINT.
        max_cells:    Cell cap for inference.

    Returns:
        edge_index: (2, E) LongTensor.
    """
    import pandas as pd

    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # ── Cache check ────────────────────────────────────────────────────────────
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_fp = _grn_cache_path(cache_dir, dataset_tag, n_genes)
        if cache_fp.exists():
            logger.info(f"[scprint_grn] cache hit: {cache_fp}")
            grn_df = pd.read_parquet(cache_fp)
            return _df_to_edge_index(grn_df, gene_to_idx, k)
    else:
        cache_fp = None

    # ── scPRINT inference ──────────────────────────────────────────────────────
    if not _scprint_available():
        logger.warning(
            "[scprint_grn] scPRINT not installed — falling back to Pearson kNN. "
            "Install with: pip install scprint"
        )
        return _pearson_fallback(adata, gene_names, k)

    try:
        grn_df = _run_scprint_inference(adata, gene_names, max_cells=max_cells, ckpt=ckpt)
    except Exception as exc:
        logger.warning(f"[scprint_grn] inference failed ({exc}) — falling back to Pearson kNN.")
        return _pearson_fallback(adata, gene_names, k)

    # Filter to gene_names universe
    gene_set = set(gene_names)
    grn_df = grn_df[
        grn_df["source"].isin(gene_set) & grn_df["target"].isin(gene_set)
    ].reset_index(drop=True)

    n_edges_raw = len(grn_df)
    if n_edges_raw == 0:
        logger.warning(
            "[scprint_grn] Zero edges after filtering to gene_names — "
            "likely gene name mismatch (ENSEMBL vs Hugo). Falling back to Pearson kNN."
        )
        return _pearson_fallback(adata, gene_names, k)

    logger.info(
        f"[scprint_grn] {n_edges_raw} raw edges → filtering to top-{k} per gene "
        f"({n_genes} genes)"
    )

    # Keep top-k outgoing edges per source gene
    grn_df = (
        grn_df.sort_values("weight", ascending=False)
        .groupby("source", as_index=False)
        .head(k)
        .reset_index(drop=True)
    )

    # Cache
    if cache_fp is not None:
        grn_df.to_parquet(cache_fp, index=False)
        logger.info(f"[scprint_grn] cached {len(grn_df)} edges → {cache_fp}")

    return _df_to_edge_index(grn_df, gene_to_idx, k)


def _df_to_edge_index(
    grn_df,
    gene_to_idx: dict,
    k: int,
) -> torch.Tensor:
    """Convert filtered GRN DataFrame to (2, E) LongTensor."""
    import pandas as pd

    # Re-apply top-k in case we loaded from cache without re-filtering
    grn_df = (
        grn_df.sort_values("weight", ascending=False)
        .groupby("source", as_index=False)
        .head(k)
        .reset_index(drop=True)
    )

    # Map gene names → integer indices
    src_idx = grn_df["source"].map(gene_to_idx)
    tgt_idx = grn_df["target"].map(gene_to_idx)

    # Drop rows where gene not in index (guard against stale cache)
    valid = src_idx.notna() & tgt_idx.notna()
    if not valid.all():
        n_drop = (~valid).sum()
        logger.warning(f"[scprint_grn] Dropping {n_drop} edges with unknown genes.")
        src_idx = src_idx[valid]
        tgt_idx = tgt_idx[valid]

    src = torch.tensor(src_idx.values.astype(int), dtype=torch.long)
    tgt = torch.tensor(tgt_idx.values.astype(int), dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)

    logger.info(f"[scprint_grn] edge_index: {edge_index.shape[1]} edges over {len(gene_to_idx)} genes")
    return edge_index


def _pearson_fallback(adata, gene_names: List[str], k: int) -> torch.Tensor:
    """Pearson-kNN fallback when scPRINT is unavailable."""
    from tsgnn.data.nature_adapter import _as_dense, _correlation_knn_edges
    import numpy as np

    X = _as_dense(adata.X)
    gene_all = list(adata.var_names.astype(str))
    gene_to_col = {g: i for i, g in enumerate(gene_all)}
    keep = [gene_to_col[g] for g in gene_names if g in gene_to_col]
    X = X[:, keep]
    return _correlation_knn_edges(X, k=k)
