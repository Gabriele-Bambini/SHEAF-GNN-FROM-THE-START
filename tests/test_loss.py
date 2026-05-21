"""Unit tests for the composite loss function."""

import pytest
import torch

from tsgnn.training.loss import TSGNNLoss, _differentiable_spearman, _soft_rank


@pytest.fixture
def criterion():
    return TSGNNLoss(lambda_1=0.1, lambda_2=0.5, lambda_3=0.01, tau=0.5)


def test_expression_loss(criterion):
    K, N, dim = 5, 10, 10
    preds = [torch.randn(N, dim) for _ in range(K)]
    targets = [torch.randn(N, dim) for _ in range(K)]
    loss = criterion.expression_loss(preds, targets)
    assert loss.shape == ()
    assert loss > 0


def test_expression_loss_zero_for_identical(criterion):
    K, N, dim = 5, 10, 10
    data = [torch.randn(N, dim) for _ in range(K)]
    loss = criterion.expression_loss(data, data)
    assert loss.item() < 1e-6


def test_topological_loss_single_step(criterion):
    laps = [torch.randn(20, 20)]
    loss = criterion.topological_preservation_loss(laps)
    assert loss.item() == 0.0  # Need at least 2 steps


def test_topological_loss_hinge(criterion):
    K = 5
    laps = [torch.zeros(20, 20) for _ in range(K)]
    # No change -> loss should be 0
    loss = criterion.topological_preservation_loss(laps)
    assert loss.item() == 0.0

    # Large change -> loss should be positive
    laps2 = [torch.randn(20, 20) * 5 for _ in range(K)]
    loss2 = criterion.topological_preservation_loss(laps2)
    assert loss2.item() > 0


def test_forward_returns_components(criterion):
    K, N, d, dim = 5, 10, 4, 10
    Nd = N * d
    preds = [torch.randn(N, dim) for _ in range(K)]
    targets = [torch.randn(N, dim) for _ in range(K)]
    laps = [torch.randn(Nd, Nd) for _ in range(K)]

    total, components = criterion(preds, targets, laps)
    assert "loss_expr" in components
    assert "loss_topo" in components
    assert "loss_total" in components
    assert total.requires_grad is False or total.grad_fn is not None


def test_soft_rank_differentiable():
    x = torch.randn(10, requires_grad=True)
    ranks = _soft_rank(x)
    ranks.sum().backward()
    assert x.grad is not None


def test_spearman_perfect_correlation():
    x = torch.arange(10, dtype=torch.float32)
    corr = _differentiable_spearman(x.unsqueeze(0), x.unsqueeze(0))
    assert corr.item() > 0.95  # Should be close to 1
