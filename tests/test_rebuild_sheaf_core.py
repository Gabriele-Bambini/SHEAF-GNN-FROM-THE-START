import pytest
import torch


def _path_edges() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [1, 2, 3, 4, 2],
        ],
        dtype=torch.long,
    )


def test_cayley_map_is_orthogonal_and_differentiable():
    from sheaf_ts.core.orthogonal import cayley_map, orthogonality_error

    torch.manual_seed(7)
    params = torch.randn(12, 6, dtype=torch.float64, requires_grad=True)

    maps = cayley_map(params, d=4)
    loss = maps.square().sum()
    loss.backward()

    assert maps.shape == (12, 4, 4)
    assert orthogonality_error(maps) < 1e-10
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_general_laplacian_matches_dense_coboundary_for_nonorthogonal_maps():
    from sheaf_ts.core.coboundary import build_dense_coboundary
    from sheaf_ts.core.laplacian_builders import build_general_laplacian

    torch.manual_seed(11)
    num_nodes = 5
    edge_index = _path_edges()
    maps = torch.randn(edge_index.shape[1], 2, 3, 3, dtype=torch.float64)
    edge_weights = torch.linspace(0.4, 1.7, edge_index.shape[1], dtype=torch.float64)

    B = build_dense_coboundary(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
    )
    expected = B.T @ B
    actual = build_general_laplacian(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
        normalize=False,
        eps=0.0,
    )

    assert torch.allclose(actual, expected, atol=1e-10)


def test_connection_laplacian_matches_general_for_orthogonal_maps():
    from sheaf_ts.core.laplacian_builders import (
        build_connection_laplacian,
        build_general_laplacian,
    )
    from sheaf_ts.core.orthogonal import cayley_map

    torch.manual_seed(13)
    num_nodes = 5
    edge_index = _path_edges()
    num_edges = edge_index.shape[1]
    d = 4
    params = torch.randn(num_edges * 2, d * (d - 1) // 2, dtype=torch.float64)
    maps = cayley_map(params, d=d).reshape(num_edges, 2, d, d)
    edge_weights = torch.linspace(0.5, 1.4, num_edges, dtype=torch.float64)

    general = build_general_laplacian(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
        normalize=False,
        eps=0.0,
    )
    connection = build_connection_laplacian(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
        normalize=False,
        eps=0.0,
        verify_orthogonal=True,
    )

    assert torch.allclose(connection, general, atol=1e-10)


def test_connection_laplacian_rejects_nonorthogonal_maps_when_verified():
    from sheaf_ts.core.laplacian_builders import build_connection_laplacian

    torch.manual_seed(17)
    edge_index = _path_edges()
    maps = torch.randn(edge_index.shape[1], 2, 3, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="orthogonal"):
        build_connection_laplacian(
            num_nodes=5,
            edge_index=edge_index,
            maps=maps,
            normalize=False,
            verify_orthogonal=True,
        )


def test_normalized_connection_laplacian_spectrum_is_bounded():
    from sheaf_ts.core.laplacian_builders import build_connection_laplacian
    from sheaf_ts.core.orthogonal import cayley_map
    from sheaf_ts.core.spectral_checks import check_normalized_spectrum

    torch.manual_seed(19)
    edge_index = _path_edges()
    num_edges = edge_index.shape[1]
    d = 3
    params = torch.randn(num_edges * 2, d * (d - 1) // 2, dtype=torch.float64)
    maps = cayley_map(params, d=d).reshape(num_edges, 2, d, d)

    laplacian = build_connection_laplacian(
        num_nodes=5,
        edge_index=edge_index,
        maps=maps,
        normalize=True,
        eps=0.0,
        verify_orthogonal=True,
    )
    report = check_normalized_spectrum(laplacian, atol=1e-9)

    assert report.min_eigenvalue >= -1e-9
    assert report.max_eigenvalue <= 2.0 + 1e-9


def test_spectral_checks_fail_closed_on_invalid_matrices():
    from sheaf_ts.core.spectral_checks import check_psd, check_symmetric

    non_symmetric = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
    indefinite = torch.tensor([[1.0, 0.0], [0.0, -0.1]], dtype=torch.float64)

    with pytest.raises(ValueError, match="symmetric"):
        check_symmetric(non_symmetric)

    with pytest.raises(ValueError, match="PSD"):
        check_psd(indefinite)
