"""Unit tests for the Neural Sheaf Diffusion layer."""

import pytest
import torch

from tsgnn.model.sheaf import SheafDiffusionLayer


@pytest.fixture
def layer():
    N, E, d = 10, 20, 4
    edge_index = torch.randint(0, N, (2, E))
    return SheafDiffusionLayer(N, E, d, edge_index)


def test_laplacian_shape(layer):
    L = layer.compute_connection_laplacian()
    Nd = layer.num_nodes * layer.d
    assert L.shape == (Nd, Nd)


def test_laplacian_symmetry(layer):
    L = layer.compute_connection_laplacian()
    assert torch.allclose(L, L.T, atol=1e-5), "Laplacian should be symmetric"


def test_identity_maps_reduce_to_graph_laplacian():
    """With identity restriction maps, sheaf Laplacian should reduce to graph Laplacian."""
    N, d = 5, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    E = len(edges)
    edge_index = torch.tensor([[u for u, v in edges], [v for u, v in edges]], dtype=torch.long)

    layer = SheafDiffusionLayer(N, E, d, edge_index, laplacian_eps=0.0)

    # Set restriction maps to identity
    with torch.no_grad():
        layer.restriction_maps.fill_(0)
        for e in range(E):
            for s in range(2):
                for i in range(d):
                    layer.restriction_maps[e, s, i, i] = 1.0

    L = layer.compute_connection_laplacian()

    # Check block structure: L_F[i*d, j*d] should match graph Laplacian
    L_block = L[::d, ::d].detach()
    # Diagonal should be degree (number of incident edges)
    assert L_block[0, 0] == 1.0  # node 0: degree 1
    assert L_block[1, 1] == 2.0  # node 1: degree 2
    assert L_block[2, 2] == 2.0  # node 2: degree 2


def test_diffusion_residual(layer):
    """Diffusion should include residual connection."""
    N, d = layer.num_nodes, layer.d
    x = torch.randn(N, d)
    L = layer.compute_connection_laplacian()
    x_out = layer.diffuse(x, L)
    assert x_out.shape == x.shape
    # With residual, output should differ from input
    assert not torch.allclose(x_out, x)


def test_forward_returns_correct_shapes(layer):
    N, d = layer.num_nodes, layer.d
    x = torch.randn(N, d)
    x_out, L = layer.forward(x, num_steps=2)
    assert x_out.shape == (N, d)
    assert L.shape == (N * d, N * d)


def test_external_maps(layer):
    """Can pass external restriction maps."""
    N, d, E = layer.num_nodes, layer.d, layer.num_edges
    ext_maps = torch.randn(E, 2, d, d)
    L = layer.compute_connection_laplacian(ext_maps)
    assert L.shape == (N * d, N * d)
