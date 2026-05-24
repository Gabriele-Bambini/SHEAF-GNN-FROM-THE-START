import torch
import pytest
from tsgnn.evaluation.geometric_metrics import (
    edge_restriction_shift,
    so_distance_matrix,
    h1_dimension_approx,
    allele_divergence,
)


def _random_orthogonal(n, d):
    """Helper: (n, d, d) random orthogonal matrices."""
    mats = []
    for _ in range(n):
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        mats.append(Q)
    return torch.stack(mats)


def test_edge_restriction_shift_zero_for_identical():
    E, d = 50, 4
    F = _random_orthogonal(E, d).unsqueeze(1).expand(E, 2, d, d)
    shifts = edge_restriction_shift(F, F)
    assert shifts.shape == (E,)
    assert torch.allclose(shifts, torch.zeros(E), atol=1e-5)


def test_edge_restriction_shift_positive_for_different():
    E, d = 50, 4
    F1 = _random_orthogonal(E, d).unsqueeze(1).expand(E, 2, d, d).clone()
    F2 = _random_orthogonal(E, d).unsqueeze(1).expand(E, 2, d, d).clone()
    shifts = edge_restriction_shift(F1, F2)
    assert (shifts > 0).all(), "All edge shifts should be positive for different maps"


def test_so_distance_matrix_shape():
    Rs = _random_orthogonal(3, 4)
    D = so_distance_matrix(Rs)
    assert D.shape == (3, 3)


def test_so_distance_matrix_zero_diagonal():
    Rs = _random_orthogonal(3, 4)
    D = so_distance_matrix(Rs)
    assert torch.allclose(D.diagonal(), torch.zeros(3), atol=1e-5)


def test_so_distance_matrix_symmetric():
    Rs = _random_orthogonal(3, 4)
    D = so_distance_matrix(Rs)
    assert torch.allclose(D, D.T, atol=1e-5)


def test_so_distance_matrix_in_range():
    Rs = _random_orthogonal(5, 4)
    D = so_distance_matrix(Rs)
    assert (D >= 0).all()
    assert (D <= 1.0 + 1e-5).all()


def test_h1_dimension_approx_identity_laplacian():
    # Identity Laplacian -> no null space -> H1 = 0
    N, d = 10, 4
    L = torch.eye(N * d)
    h1 = h1_dimension_approx(L)
    assert h1 == 0, f"Expected H1=0 for identity, got {h1}"


def test_h1_dimension_approx_zero_laplacian():
    # Zero Laplacian -> fully degenerate -> H1 = N*d
    N, d = 5, 4
    L = torch.zeros(N * d, N * d)
    h1 = h1_dimension_approx(L)
    assert h1 == N * d


def test_allele_divergence_zero_for_same():
    N = 100
    pred = torch.randn(N)
    div = allele_divergence(pred, pred)
    assert abs(div) < 1e-6


def test_allele_divergence_positive_for_different():
    N = 100
    pred_wt = torch.randn(N)
    pred_mut = pred_wt + torch.randn(N) * 2.0
    div = allele_divergence(pred_wt, pred_mut)
    assert div > 0
