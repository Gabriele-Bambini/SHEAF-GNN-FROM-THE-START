"""Tests for cross-dataset gene intersection and zero-shot transfer."""
import numpy as np
import pytest
import torch

from tsgnn.data.nature_adapter import intersect_gene_sets, _correlation_knn_edges


def _make_bundle(gene_names, K=5, input_dim=1, seed=0):
    """Create a synthetic bundle for testing."""
    rng = np.random.default_rng(seed)
    N = len(gene_names)
    X = rng.standard_normal((K, N, input_dim)).astype(np.float32)
    X_seq = torch.tensor(X)
    edge_index = _correlation_knn_edges(X.mean(axis=-1), k=min(4, N - 1))
    return {
        "X_seq": X_seq,
        "edge_index": edge_index,
        "gene_names": gene_names,
        "allele_label": "WT",
    }


class TestIntersectGeneSets:
    def test_basic_intersection(self):
        genes_a = [f"GENE{i}" for i in range(300)]
        genes_b = [f"GENE{i}" for i in range(50, 350)]
        src = _make_bundle(genes_a, seed=1)
        tgt = _make_bundle(genes_b, seed=2)
        s, t, common = intersect_gene_sets(src, tgt, min_genes=200)
        assert len(common) == 250
        assert s["gene_names"] == t["gene_names"] == common
        assert s["X_seq"].shape[1] == len(common)
        assert t["X_seq"].shape[1] == len(common)

    def test_no_duplicates(self):
        genes_a = [f"GENE{i}" for i in range(300)]
        genes_b = [f"GENE{i}" for i in range(100, 400)]
        src = _make_bundle(genes_a, seed=3)
        tgt = _make_bundle(genes_b, seed=4)
        s, t, common = intersect_gene_sets(src, tgt, min_genes=100)
        assert len(set(common)) == len(common)

    def test_too_small_intersection_raises(self):
        genes_a = [f"A{i}" for i in range(300)]
        genes_b = [f"B{i}" for i in range(300)]
        # Only overlap on nothing
        src = _make_bundle(genes_a, seed=5)
        tgt = _make_bundle(genes_b, seed=6)
        with pytest.raises(ValueError, match="Gene intersection too small"):
            intersect_gene_sets(src, tgt, min_genes=200)

    def test_edges_rebuilt(self):
        genes = [f"GENE{i}" for i in range(300)]
        src = _make_bundle(genes, seed=7)
        tgt = _make_bundle(genes, seed=8)
        s, t, common = intersect_gene_sets(src, tgt, min_genes=200)
        # Edges should be fresh (from subsetted correlations)
        assert s["edge_index"].shape[0] == 2
        assert t["edge_index"].shape[0] == 2
        assert s["edge_index"].max() < len(common)
        assert t["edge_index"].max() < len(common)

    def test_gene_ordering_consistent(self):
        genes_a = [f"G{i:03d}" for i in range(250)]
        genes_b = [f"G{i:03d}" for i in range(50, 300)]
        src = _make_bundle(genes_a, seed=9)
        tgt = _make_bundle(genes_b, seed=10)
        s, t, common = intersect_gene_sets(src, tgt, min_genes=100)
        assert common == sorted(common)
        # Values should correspond to correct gene positions
        src_idx = {g: i for i, g in enumerate(genes_a)}
        for i, g in enumerate(common):
            orig_i = src_idx[g]
            assert torch.allclose(s["X_seq"][:, i, :], src["X_seq"][:, orig_i, :])
