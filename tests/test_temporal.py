"""Unit tests for the GRU temporal evolution module."""

import pytest
import torch

from tsgnn.model.temporal import TemporalRestrictionEvolution


@pytest.fixture
def module():
    return TemporalRestrictionEvolution(stalk_dim=4)


def test_output_shape(module):
    E, N, d = 50, 20, 4
    maps = torch.randn(E, 2, d, d)
    features = torch.randn(N, d)
    edge_index = torch.randint(0, N, (2, E))
    new_maps = module(maps, features, edge_index)
    assert new_maps.shape == maps.shape


def test_different_from_input(module):
    E, N, d = 50, 20, 4
    maps = torch.randn(E, 2, d, d)
    features = torch.randn(N, d)
    edge_index = torch.randint(0, N, (2, E))
    new_maps = module(maps, features, edge_index)
    assert not torch.allclose(new_maps, maps)


def test_gradient_flow(module):
    E, N, d = 10, 5, 4
    maps = torch.randn(E, 2, d, d, requires_grad=True)
    features = torch.randn(N, d)
    edge_index = torch.randint(0, N, (2, E))
    new_maps = module(maps, features, edge_index)
    loss = new_maps.sum()
    loss.backward()
    assert maps.grad is not None
    assert maps.grad.norm() > 0


def test_paga_pseudotime_no_infs():
    """PAGA pseudotime must not contain infinite values."""
    import numpy as np
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    np.random.seed(42)
    X = np.random.rand(100, 50)
    adata = ad.AnnData(X)
    sc.pp.neighbors(adata)
    try:
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    except (TypeError, ImportError):
        try:
            sc.tl.leiden(adata)
        except ImportError:
            adata.obs["leiden"] = "0"
    from tsgnn.data.temporal import compute_pseudotime
    pt = compute_pseudotime(adata, method="paga")
    assert not np.any(np.isinf(pt)), "Pseudotime PAGA contiene inf"
    assert not np.any(np.isnan(pt)), "Pseudotime PAGA contiene nan"
    assert pt.min() >= 0 and pt.max() <= 1
