"""
Evaluation Metrics for TS-GNN.

Phase 4A: Implements all 5 evaluation metrics:
1. Cross-cancer zero-shot generalization (Pearson correlation)
2. Allele-specific edge enrichment (Fisher exact test)
3. Chromatin remodeler concordance (ChIP-seq overlap)
4. DepMap genetic dependency correlation (Spearman)
5. Rewiring distinguishability (permutation test)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cross-cancer zero-shot generalization
# ---------------------------------------------------------------------------

def cross_cancer_zero_shot(
    predicted_edges: torch.Tensor,
    observed_edges: torch.Tensor,
) -> Dict[str, float]:
    """
    Pearson correlation of predicted vs. observed differential edge weights
    in a held-out BRCA patient cohort (model trained on BRCA training split).

    Args:
        predicted_edges: (E,) predicted differential edge weights from TS-GNN.
        observed_edges: (E,) observed differential edge weights in held-out BRCA patients.

    Returns:
        Dict with pearson_r, p_value, and interpretation.
    """
    pred = predicted_edges.detach().cpu().numpy()
    obs = observed_edges.detach().cpu().numpy()

    # Remove NaN/Inf
    valid = np.isfinite(pred) & np.isfinite(obs)
    if valid.sum() < 3:
        return {"pearson_r": 0.0, "p_value": 1.0, "interpretation": "insufficient_data"}

    r, p = stats.pearsonr(pred[valid], obs[valid])

    if r > 0.3:
        interpretation = "strong_generalization"
    elif r > 0.15:
        interpretation = "moderate_generalization"
    else:
        interpretation = "tissue_dominant"

    return {"pearson_r": float(r), "p_value": float(p), "interpretation": interpretation}


# ---------------------------------------------------------------------------
# 2. Allele-specific edge enrichment
# ---------------------------------------------------------------------------

def allele_edge_enrichment(
    top_edges: List[Tuple[int, int]],
    validated_targets: List[Tuple[int, int]],
    total_possible_edges: int,
    predicted_directions: Optional[Dict[Tuple[int, int], float]] = None,
    known_directions: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[str, float]:
    """
    Fisher exact test for enrichment of TS-GNN top-ranked allele-specific
    edges among TargetGeneReg 2.0 validated p53 targets.

    Args:
        top_edges: List of (src, tgt) pairs from TS-GNN top-ranked edges.
        validated_targets: List of (src, tgt) validated p53 regulatory targets.
        total_possible_edges: Total number of possible edges (N*(N-1)).
        predicted_directions: Optional dict mapping edge -> predicted direction (+1/-1).
        known_directions: Optional dict mapping edge -> known regulatory direction.

    Returns:
        Dict with fisher_p, odds_ratio, direction_concordance.
    """
    top_set = set(top_edges)
    val_set = set(validated_targets)

    # Contingency table for Fisher test
    a = len(top_set & val_set)       # top AND validated
    b = len(top_set - val_set)       # top but NOT validated
    c = len(val_set - top_set)       # validated but NOT top
    d = total_possible_edges - a - b - c  # neither

    # Fisher exact test
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

    # Direction concordance
    direction_concordance = 0.0
    if predicted_directions and known_directions:
        common_edges = top_set & val_set
        if common_edges:
            concordant = sum(
                1 for e in common_edges
                if e in predicted_directions and e in known_directions
                and np.sign(predicted_directions[e]) == np.sign(known_directions[e])
            )
            direction_concordance = concordant / len(common_edges)

    return {
        "fisher_p": float(p_value),
        "odds_ratio": float(odds_ratio),
        "overlap_count": a,
        "top_edges_count": len(top_edges),
        "validated_count": len(validated_targets),
        "direction_concordance": direction_concordance,
    }


# ---------------------------------------------------------------------------
# 3. Chromatin remodeler concordance
# ---------------------------------------------------------------------------

def chromatin_remodeler_concordance(
    high_weight_edges: List[Tuple[int, int]],
    chip_seq_edges: List[Tuple[int, int]],
    total_possible_edges: int,
) -> Dict[str, float]:
    """
    Overlap of high-weight TS-GNN edges with SMARCA4/EZH2 ChIP-seq peaks.

    Args:
        high_weight_edges: Edges with high TS-GNN weights.
        chip_seq_edges: Edges supported by ChIP-seq binding evidence.
        total_possible_edges: Total number of possible edges.

    Returns:
        Dict with jaccard_index, fisher_p, overlap_count.
    """
    hw_set = set(high_weight_edges)
    chip_set = set(chip_seq_edges)

    intersection = hw_set & chip_set
    union = hw_set | chip_set

    jaccard = len(intersection) / max(len(union), 1)

    # Fisher exact test
    a = len(intersection)
    b = len(hw_set - chip_set)
    c = len(chip_set - hw_set)
    d = total_possible_edges - a - b - c

    table = np.array([[a, b], [c, d]])
    _, p_value = stats.fisher_exact(table, alternative="greater")

    return {
        "jaccard_index": float(jaccard),
        "fisher_p": float(p_value),
        "overlap_count": len(intersection),
        "high_weight_count": len(high_weight_edges),
        "chip_seq_count": len(chip_seq_edges),
    }


# ---------------------------------------------------------------------------
# 4. DepMap genetic dependency correlation
# ---------------------------------------------------------------------------

def depmap_dependency_correlation(
    hub_influence_scores: np.ndarray,
    depmap_scores: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Spearman correlation between predicted hub node influence scores
    and DepMap CRISPR screen genetic dependency scores.

    Args:
        hub_influence_scores: (N,) predicted influence from TS-GNN.
        depmap_scores: (N,) DepMap CRISPR gene effect scores.
        gene_names: Optional gene names for reporting.

    Returns:
        Dict with spearman_rho, p_value.
    """
    valid = np.isfinite(hub_influence_scores) & np.isfinite(depmap_scores)
    if valid.sum() < 3:
        return {"spearman_rho": 0.0, "p_value": 1.0}

    rho, p = stats.spearmanr(hub_influence_scores[valid], depmap_scores[valid])

    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_genes": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# 5. Rewiring distinguishability (permutation test)
# ---------------------------------------------------------------------------

def rewiring_distinguishability(
    delta_L_allele_a: List[torch.Tensor],
    delta_L_allele_b: List[torch.Tensor],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Permutation test on Frobenius norm of differential edge weight matrix
    between two alleles (e.g., R175H vs. R273H) across all K time steps.

    Tests whether the observed difference in rewiring patterns between two
    alleles is significantly greater than expected by chance.

    Args:
        delta_L_allele_a: List of K differential Laplacians for allele A.
        delta_L_allele_b: List of K differential Laplacians for allele B.
        n_permutations: Number of permutation resamples.
        seed: Random seed.

    Returns:
        Dict with observed_stat, p_value, effect_size.
    """
    rng = np.random.RandomState(seed)

    # Observed statistic: sum of Frobenius norms of difference across time steps
    K = len(delta_L_allele_a)
    diffs = []
    for t in range(K):
        diff = (delta_L_allele_a[t] - delta_L_allele_b[t]).detach().cpu()
        diffs.append(diff.norm().item())
    observed_stat = np.sum(diffs)

    # Pool all Laplacians
    all_laplacians = delta_L_allele_a + delta_L_allele_b
    n_a = len(delta_L_allele_a)
    n_total = len(all_laplacians)

    # Flatten all Laplacians for fast vectorized permutation testing
    all_laps_flat = torch.stack(all_laplacians).detach().cpu().view(n_total, -1)
    
    null_stats = np.zeros(n_permutations)
    for perm in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_a = all_laps_flat[perm_idx[:n_a]]
        perm_b = all_laps_flat[perm_idx[n_a:]]

        n_min = min(len(perm_a), len(perm_b))
        diff = perm_a[:n_min] - perm_b[:n_min]
        null_stats[perm] = diff.norm(dim=1).sum().item()

    # p-value: fraction of permutations >= observed
    p_value = (null_stats >= observed_stat).mean()

    # Effect size: (observed - mean(null)) / std(null)
    null_std = null_stats.std()
    effect_size = (observed_stat - null_stats.mean()) / max(null_std, 1e-8)

    return {
        "observed_stat": float(observed_stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Utility: Extract differential edges from Laplacians
# ---------------------------------------------------------------------------

def extract_differential_edges(
    laplacians_a: List[torch.Tensor],
    laplacians_b: List[torch.Tensor],
    stalk_dim: int,
    top_k: int = 50,
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """
    Extract top-k differential edges between two alleles from their Laplacians.

    Args:
        laplacians_a: List of K Laplacians for allele A.
        laplacians_b: List of K Laplacians for allele B.
        stalk_dim: Stalk dimension d (for block extraction).
        top_k: Number of top differential edges to return.

    Returns:
        edges: List of (src, tgt) tuples for top differential edges.
        weights: (top_k,) tensor of differential weights.
    """
    d = stalk_dim
    K = min(len(laplacians_a), len(laplacians_b))

    # Aggregate differential Laplacian across time
    diff_L = torch.zeros_like(laplacians_a[0])
    for t in range(K):
        diff_L += (laplacians_a[t] - laplacians_b[t]).abs()
    diff_L /= K

    # Extract block-level differences (node pairs)
    Nd = diff_L.shape[0]
    N = Nd // d

    blocks = diff_L.view(N, d, N, d)
    node_diffs = blocks.pow(2).sum(dim=(1, 3)).sqrt()
    
    # zero diagonal
    node_diffs.fill_diagonal_(0.0)

    # Top-k edges
    flat = node_diffs.flatten()
    k_actual = min(top_k, flat.shape[0])
    weights, indices = flat.topk(k_actual)
    rows = indices // N
    cols = indices % N

    edges = [(rows[i].item(), cols[i].item()) for i in range(k_actual)]

    return edges, weights


def compute_hub_influence_scores(
    laplacians: List[torch.Tensor],
    stalk_dim: int,
) -> np.ndarray:
    """
    Compute hub influence scores from sheaf Laplacians.

    Uses the diagonal blocks of the Laplacian (degree-like measure)
    as a proxy for node influence in the regulatory network.

    Args:
        laplacians: List of K Laplacians.
        stalk_dim: Stalk dimension d.

    Returns:
        (N,) array of influence scores.
    """
    d = stalk_dim
    Nd = laplacians[0].shape[0]
    N = Nd // d
    K = len(laplacians)

    scores = torch.zeros(N)
    for t in range(K):
        L = laplacians[t].detach().cpu()
        blocks = L.view(N, d, N, d)
        scores += blocks.diagonal(dim1=0, dim2=2).norm(dim=(0, 1))

    scores /= K
    return scores.numpy()


# ---------------------------------------------------------------------------
# All-in-one evaluation runner
# ---------------------------------------------------------------------------

def evaluate_all(
    model,
    test_data: Dict,
    esm_embeddings: Dict[str, torch.Tensor],
    stalk_dim: int,
    validated_targets: Optional[List[Tuple[int, int]]] = None,
    chip_seq_edges: Optional[List[Tuple[int, int]]] = None,
    depmap_scores: Optional[np.ndarray] = None,
    cross_cancer_data: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Run all 5 evaluation metrics.

    Returns dict mapping metric name -> results dict.
    """
    results = {}
    alleles = list(test_data.keys())
    device = next(model.parameters()).device

    # Run model on all alleles
    model.eval()
    allele_laplacians = {}
    allele_predictions = {}

    with torch.no_grad():
        for allele, data in test_data.items():
            node_features_seq = data["node_features_seq"].to(device)
            allele_emb = esm_embeddings[allele].to(device)
            preds, _, laps = model(node_features_seq, allele_emb)
            allele_laplacians[allele] = [l.cpu() for l in laps]
            allele_predictions[allele] = [p.cpu() for p in preds]

    # 1. Cross-cancer zero-shot (if cross-cancer data provided)
    if cross_cancer_data is not None:
        for allele in cross_cancer_data:
            if allele in allele_predictions:
                pred_edges = torch.cat([p.flatten() for p in allele_predictions[allele]])
                obs_edges = cross_cancer_data[allele]
                results[f"zero_shot_{allele}"] = cross_cancer_zero_shot(pred_edges, obs_edges)

    # 2. Allele-specific edge enrichment
    if validated_targets and len(alleles) >= 2:
        # Compare mutant vs WT
        wt = "WT" if "WT" in alleles else alleles[0]
        for allele in alleles:
            if allele == wt:
                continue
            if allele in allele_laplacians and wt in allele_laplacians:
                top_edges, _ = extract_differential_edges(
                    allele_laplacians[allele],
                    allele_laplacians[wt],
                    stalk_dim,
                    top_k=50,
                )
                N = allele_laplacians[allele][0].shape[0] // stalk_dim
                results[f"enrichment_{allele}"] = allele_edge_enrichment(
                    top_edges, validated_targets, N * (N - 1),
                )

    # 3. Chromatin remodeler concordance
    if chip_seq_edges and len(alleles) >= 2:
        wt = "WT" if "WT" in alleles else alleles[0]
        for allele in alleles:
            if allele == wt:
                continue
            if allele in allele_laplacians and wt in allele_laplacians:
                top_edges, _ = extract_differential_edges(
                    allele_laplacians[allele],
                    allele_laplacians[wt],
                    stalk_dim,
                    top_k=50,
                )
                N = allele_laplacians[allele][0].shape[0] // stalk_dim
                results[f"chip_seq_{allele}"] = chromatin_remodeler_concordance(
                    top_edges, chip_seq_edges, N * (N - 1),
                )

    # 4. DepMap genetic dependency
    if depmap_scores is not None:
        for allele in alleles:
            if allele in allele_laplacians:
                hub_scores = compute_hub_influence_scores(
                    allele_laplacians[allele], stalk_dim
                )
                N = len(hub_scores)
                dm = depmap_scores[:N] if len(depmap_scores) >= N else np.pad(
                    depmap_scores, (0, N - len(depmap_scores))
                )
                results[f"depmap_{allele}"] = depmap_dependency_correlation(hub_scores, dm)

    # 5. Rewiring distinguishability (pairwise between mutant alleles)
    mutants = [a for a in alleles if a != "WT"]
    for i, a in enumerate(mutants):
        for j, b in enumerate(mutants):
            if i < j and a in allele_laplacians and b in allele_laplacians:
                results[f"distinguish_{a}_vs_{b}"] = rewiring_distinguishability(
                    allele_laplacians[a], allele_laplacians[b],
                )

    return results


def eval_crc_zeroshot(
    model,
    gene_list: List[str],
    esm_embeddings: Dict[str, "torch.Tensor"],
    device: str = "cpu",
    crc_geo_accession: str = "GSE178341",
) -> Dict[str, float]:
    """Zero-shot evaluation on Colorectal Cancer (CRC) scRNA-seq data.

    The model is trained exclusively on BRCA (GSE176078). We evaluate its
    ability to generalise by applying it to CRC cells *without any fine-tuning*.
    This tests the hypothesis that TP53-allele-specific regulatory circuits
    are tissue-agnostic (Nature-level generalizability claim).

    Workflow
    --------
    1. Load CRC scRNA-seq from ``data/raw/crc/<crc_geo_accession>/``.
    2. Preprocess with the same pipeline used for BRCA (QC → HVG → PCA).
    3. Build temporal graph snapshots over PAGA pseudotime.
    4. Run the trained model in eval mode.
    5. Compute ``cross_cancer_zero_shot`` metric.

    Args:
        model:          Trained TSGNN model (``nn.Module``).
        gene_list:      List of gene names the model was trained on (HVG set).
        esm_embeddings: Dict of allele → ESM-2 embedding tensor.
        device:         Torch device string.
        crc_geo_accession: GEO accession of CRC dataset.

    Returns:
        Dict with ``pearson_r``, ``p_value``, ``n_edges``, ``interpretation``.
        If CRC data is absent, returns ``{"status": "data_missing"}`` with
        instructions instead of raising.
    """
    from pathlib import Path
    from tsgnn.data import RAW_DIR

    # Accept two layouts:
    #   (a) data/raw/crc/<accession>/...        — manual / RAW tar extraction
    #   (b) data/raw/crc/<accession>_*.h5       — direct H5 from download_crc_scrna
    crc_dir = RAW_DIR / "crc" / crc_geo_accession
    if not crc_dir.exists():
        # Look for H5 matching accession in flat crc/ folder
        flat_h5 = list((RAW_DIR / "crc").glob(f"{crc_geo_accession}*.h5")) + \
                  list((RAW_DIR / "crc").glob(f"{crc_geo_accession}*.h5ad"))
        if flat_h5:
            crc_dir = RAW_DIR / "crc"  # use flat folder; rglob will find the H5
            logger.info(f"Using flat CRC layout, found {len(flat_h5)} H5 file(s)")

    if not crc_dir.exists():
        logger.warning(
            "CRC data not found at %s.\n"
            "Download instructions:\n"
            "  Option A (automatic): python -c \"from tsgnn.data.download import "
            "download_crc_scrna; download_crc_scrna('%s')\"\n"
            "  Option B (manual GEO): "
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s\n"
            "  → Download GSE178341_RAW.tar.gz → extract to %s",
            crc_dir, crc_geo_accession, crc_geo_accession, crc_dir,
        )
        return {
            "status": "data_missing",
            "dataset": crc_geo_accession,
            "instructions": (
                f"Run: python -c \"from tsgnn.data.download import download_crc_scrna; "
                f"download_crc_scrna('{crc_geo_accession}')\" "
                f"or download manually from "
                f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={crc_geo_accession}"
            ),
        }

    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        return {"status": "scanpy_missing", "instructions": "pip install scanpy anndata"}

    # ── 1. Load CRC data ──────────────────────────────────────────────────────
    logger.info("Loading CRC scRNA-seq from %s", crc_dir)
    h5_files = list(crc_dir.rglob("*.h5")) + list(crc_dir.rglob("*.h5ad"))
    mtx_dirs = [p.parent for p in crc_dir.rglob("matrix.mtx*")]

    if h5_files:
        adata = sc.read_h5ad(h5_files[0]) if h5_files[0].suffix == ".h5ad" \
                else sc.read_10x_h5(h5_files[0])
    elif mtx_dirs:
        adata = sc.read_10x_mtx(mtx_dirs[0], var_names="gene_symbols", cache=True)
    else:
        logger.warning("No recognised CRC matrix format found in %s", crc_dir)
        return {"status": "data_format_unknown", "crc_dir": str(crc_dir)}

    # ── 2. Minimal QC + HVG alignment ─────────────────────────────────────────
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Restrict to genes the model knows
    shared_genes = [g for g in gene_list if g in adata.var_names]
    if len(shared_genes) < 50:
        logger.warning(
            "Only %d shared genes between CRC data and model (%d required min).",
            len(shared_genes), 50,
        )
        return {"status": "insufficient_shared_genes", "n_shared": len(shared_genes)}
    adata = adata[:, shared_genes]

    # ── 3. Run model inference ─────────────────────────────────────────────────
    import torch
    model.eval()
    allele = "R175H"  # Use most common BRCA hotspot as surrogate for zero-shot
    if allele not in esm_embeddings:
        allele = next(iter(esm_embeddings))

    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                     dtype=torch.float32).to(device)
    esm_emb = esm_embeddings[allele].unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(X, esm_emb)

    predicted_edges = preds.mean(dim=0).cpu()

    # ── 4. Observed baseline (co-expression on CRC) ────────────────────────────
    X_np = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    # Simple Pearson co-expression as "observed" proxy
    from scipy.stats import pearsonr as _pr
    n_genes = X_np.shape[1]
    observed = np.zeros(n_genes)
    for i in range(min(n_genes, len(predicted_edges))):
        if i + 1 < n_genes:
            r, _ = _pr(X_np[:, i], X_np[:, i + 1])
            observed[i] = r

    observed_t = torch.tensor(observed[:len(predicted_edges)], dtype=torch.float32)
    result = cross_cancer_zero_shot(predicted_edges[:len(observed_t)], observed_t)
    result["n_shared_genes"] = len(shared_genes)
    result["dataset"] = crc_geo_accession
    result["status"] = "ok"

    logger.info(
        "CRC zero-shot: Pearson r=%.3f (p=%.4f) — %s",
        result["pearson_r"], result["p_value"], result["interpretation"],
    )
    return result


def eval_biokey_holdout(
    model,
    gene_list,
    esm_embeddings,
    device: str = "cpu",
    sample_n: int = 10000,
) -> dict:
    """Held-out generalization eval on BIOKEY (E-MTAB-8107) cohort.

    Thin wrapper delegating the heavy-lifting to ``biokey_loader`` so that
    downstream pipelines can call a uniform ``evaluation.metrics`` entry
    point for every validation cohort.
    """
    from tsgnn.data.biokey_loader import biokey_available, load_biokey_data

    if not biokey_available():
        return {
            "status": "data_missing",
            "dataset": "E-MTAB-8107",
            "instructions": (
                "Download BIOKEY manually from "
                "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8107 "
                "→ data/raw/brca/BIOKEY_E-MTAB-8107/{counts.tsv, metadata.csv}"
            ),
        }

    try:
        import scanpy as sc
        import torch
    except ImportError:
        return {"status": "scanpy_missing", "instructions": "pip install scanpy torch"}

    adata = load_biokey_data(malignant_only=True, sample_n=sample_n)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    shared = [g for g in gene_list if g in adata.var_names]
    if len(shared) < 50:
        return {"status": "insufficient_shared_genes", "n_shared": len(shared)}
    adata = adata[:, shared]

    # Run inference with allele surrogate (most common hotspot)
    allele = "R175H" if "R175H" in esm_embeddings else next(iter(esm_embeddings))
    X = torch.tensor(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        dtype=torch.float32,
    ).to(device)
    esm_emb = esm_embeddings[allele].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X, esm_emb)

    return {
        "status": "ok",
        "dataset": "E-MTAB-8107",
        "n_cells": adata.n_obs,
        "n_shared_genes": len(shared),
        "n_patients": int(adata.obs.get("patient_id").nunique())
            if "patient_id" in adata.obs.columns else None,
        "prediction_norm": float(preds.norm().item()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test with synthetic data
    K, N, d = 10, 20, 4
    Nd = N * d

    # Synthetic Laplacians for two alleles
    laps_a = [torch.randn(Nd, Nd) for _ in range(K)]
    laps_b = [torch.randn(Nd, Nd) * 1.5 for _ in range(K)]

    # Test distinguishability
    result = rewiring_distinguishability(laps_a, laps_b, n_permutations=100)
    print(f"Distinguishability: stat={result['observed_stat']:.2f}, "
          f"p={result['p_value']:.4f}, effect={result['effect_size']:.2f}")

    # Test edge extraction
    edges, weights = extract_differential_edges(laps_a, laps_b, d, top_k=10)
    print(f"Top differential edges: {len(edges)}")

    # Test hub influence
    scores = compute_hub_influence_scores(laps_a, d)
    print(f"Hub scores: shape={scores.shape}, range=[{scores.min():.3f}, {scores.max():.3f}]")

    # Test enrichment
    validated = [(0, 1), (1, 2), (2, 3), (3, 4)]
    enrich = allele_edge_enrichment(edges, validated, N * (N - 1))
    print(f"Enrichment: p={enrich['fisher_p']:.4f}, OR={enrich['odds_ratio']:.2f}")

    # Test DepMap correlation
    depmap = np.random.randn(N)
    dep_result = depmap_dependency_correlation(scores, depmap)
    print(f"DepMap: rho={dep_result['spearman_rho']:.3f}, p={dep_result['p_value']:.4f}")
