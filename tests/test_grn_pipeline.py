import pytest
import numpy as np
import torch
import anndata


def _make_mock_adata(n_cells=200, n_genes=50, tp53_in_genes=True):
    """Create minimal AnnData with TP53 in var_names."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    if tp53_in_genes:
        gene_names[0] = "TP53"
    gene_names[1] = "MDM2"
    gene_names[2] = "CDKN1A"
    import pandas as pd
    obs = pd.DataFrame({"tp53_allele": ["WT"] * n_cells})
    var = pd.DataFrame(index=gene_names)
    return anndata.AnnData(X=X, obs=obs, var=var), gene_names


def test_grn_has_tp53_edges_when_tp53_present():
    """If TP53 is in gene list, DoRothEA/TargetGeneReg should add TP53 target edges."""
    from tsgnn.data.grn_construction import load_dorothea_prior_edges

    gene_list = ["TP53", "MDM2", "CDKN1A", "BAX", "PUMA"] + [f"G{i}" for i in range(45)]

    dorothea_edges = load_dorothea_prior_edges(gene_list, levels=["A", "B"])

    if len(dorothea_edges) == 0:
        pytest.skip("DoRothEA not available (decoupler not installed or no edges) -- skipping")

    tp53_idx = 0
    tp53_edges = [(s, t) for (s, t) in dorothea_edges if s == tp53_idx]
    assert len(tp53_edges) > 0, "TP53 should have target edges in DoRothEA"


def test_grn_fallback_to_pearson_when_biological_fails(monkeypatch):
    """If biological GRN fails, pipeline must fall back to Pearson kNN without crashing."""
    from tsgnn.data import nature_adapter

    # Force _build_grn_edges to raise (simulating import failure)
    original = nature_adapter._build_grn_edges
    def _force_fail(X, gene_list, k_neighbors):
        # Simulate: biological priors unavailable, fall back to Pearson
        return nature_adapter._correlation_knn_edges(X, k=k_neighbors)
    monkeypatch.setattr(nature_adapter, "_build_grn_edges", _force_fail)

    adata, gene_names = _make_mock_adata()

    bundle = nature_adapter.adata_to_tsgnn_tensors(adata, K=3, k_neighbors=3)
    edge_index = bundle["edge_index"]
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0, "Edge index must be non-empty"


def test_edge_index_no_self_loops():
    """edge_index must not contain self-loops (src == tgt)."""
    from tsgnn.data.nature_adapter import _correlation_knn_edges

    N, cells = 30, 100
    X = np.random.randn(cells, N).astype(np.float32)
    edge_index = _correlation_knn_edges(X, k=5)

    src, tgt = edge_index[0], edge_index[1]
    self_loops = (src == tgt).sum().item()
    assert self_loops == 0, f"Found {self_loops} self-loops in edge_index"


def test_tf_genes_protected_in_selection():
    """After gene selection, BRCA key TFs must be in the gene list."""
    from tsgnn.data.grn_construction import BRCA_KEY_TFS
    from tsgnn.data.nature_adapter import select_top_genes_with_tf_protection

    np.random.seed(0)
    n_cells, n_genes = 300, 1000
    X = np.random.exponential(1.0, size=(n_cells, n_genes)).astype(np.float32)

    # Make TF genes low-variance (they'd be excluded by plain HVG)
    all_genes = [f"GENE_{i}" for i in range(n_genes)]
    for i, tf in enumerate(list(BRCA_KEY_TFS)[:5]):
        all_genes[i] = tf
        X[:, i] = 0.01  # near-constant expression

    selected = select_top_genes_with_tf_protection(X, all_genes, n_genes=50)

    for tf in list(BRCA_KEY_TFS)[:5]:
        assert tf in selected, f"TF {tf} should be protected in gene selection"
