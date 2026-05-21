"""Tests for the vectorized sheaf Laplacian implementation."""
import pytest
import torch
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion


@pytest.fixture
def vec_layer():
    torch.manual_seed(42)
    N, E, d = 10, 25, 3
    edge_index = torch.randint(0, N, (2, E))
    return VectorizedSheafDiffusion(N, E, d, edge_index)


def test_laplacian_shape(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    assert L.shape == (vec_layer.Nd, vec_layer.Nd)


def test_laplacian_symmetry(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    assert torch.allclose(L, L.T, atol=1e-5)


def test_laplacian_psd(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    eigs = torch.linalg.eigvalsh(L)
    assert (eigs >= -1e-5).all()


def test_spectral_gap_positive(vec_layer):
    gap = vec_layer.spectral_gap()
    assert gap >= 0, f"Spectral gap should be non-negative, got {gap}"


def test_condition_number_finite(vec_layer):
    kappa = vec_layer.condition_number()
    assert kappa > 0 and kappa < 1e10


def test_verify_properties_dict(vec_layer):
    props = vec_layer.verify_laplacian_properties()
    assert props["is_symmetric"] is True
    assert props["is_psd"] is True
    assert props["spectral_gap"] >= 0


def test_forward_shapes(vec_layer):
    N, d = vec_layer.num_nodes, vec_layer.d
    x = torch.randn(N, d)
    out, L = vec_layer.forward(x, num_steps=2)
    assert out.shape == (N, d)
    assert L.shape == (N * d, N * d)


def test_external_maps(vec_layer):
    E, d = vec_layer.num_edges, vec_layer.d
    ext = torch.randn(E, 2, d, d)
    L = vec_layer.compute_connection_laplacian_vectorized(ext)
    assert L.shape == (vec_layer.Nd, vec_layer.Nd)
