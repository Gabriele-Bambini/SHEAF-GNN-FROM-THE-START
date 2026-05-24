"""Tests for GRN construction module."""
import pytest
import torch
from tsgnn.data.grn_construction import construct_base_grn, _generate_synthetic_grn


def test_construct_base_grn_raises_without_data():
    """construct_base_grn must raise RuntimeError if no regulatory databases are found."""
    gene_list = [f"GENE_{i}" for i in range(50)]
    gene_list[0] = "TP53"
    with pytest.raises(RuntimeError, match="No prior regulatory edges"):
        construct_base_grn(gene_list)


def test_synthetic_grn_dimensions():
    """_generate_synthetic_grn returns correct edge format and no self-loops."""
    N = 50
    target_edges = 200
    edges = _generate_synthetic_grn(N, target_edges=target_edges)
    assert isinstance(edges, dict)
    assert len(edges) > 0
    # No self-loops
    for (src, tgt) in edges:
        assert src != tgt, f"Self-loop at node {src}"
    # All indices in valid range
    for (src, tgt) in edges:
        assert 0 <= src < N and 0 <= tgt < N


def test_no_self_loops():
    """Synthetic GRN must not contain self-loops."""
    gene_list = [f"GENE_{i}" for i in range(20)]
    edges = _generate_synthetic_grn(len(gene_list), target_edges=100)
    for (src, tgt) in edges:
        assert src != tgt, "Self-loop found in GRN"
