"""Orthogonal map parametrizations for sheaf restriction maps."""

from __future__ import annotations

import torch

__all__ = [
    "cayley_map",
    "matrix_exp_map",
    "orthogonality_error",
    "skew_from_params",
]


def _param_dim(d: int) -> int:
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    return d * (d - 1) // 2


def skew_from_params(params: torch.Tensor, d: int) -> torch.Tensor:
    """Build skew-symmetric matrices from compact lower-triangle parameters.

    Args:
        params: tensor with final dimension ``d * (d - 1) // 2``.
        d: matrix size.

    Returns:
        Tensor with shape ``params.shape[:-1] + (d, d)`` and ``A.T == -A``.
    """
    expected = _param_dim(d)
    if params.shape[-1] != expected:
        raise ValueError(
            f"expected last dimension {expected} for d={d}, got {params.shape[-1]}"
        )

    matrix = torch.zeros(
        *params.shape[:-1],
        d,
        d,
        dtype=params.dtype,
        device=params.device,
    )
    row, col = torch.tril_indices(d, d, offset=-1, device=params.device)
    matrix[..., row, col] = params
    return matrix - matrix.transpose(-1, -2)


def cayley_map(params: torch.Tensor, d: int) -> torch.Tensor:
    """Map skew parameters to orthogonal matrices via the Cayley transform."""
    skew = skew_from_params(params, d)
    eye = torch.eye(d, dtype=params.dtype, device=params.device)
    left = eye - 0.5 * skew
    right = eye + 0.5 * skew
    return torch.linalg.solve(left, right)


def matrix_exp_map(params: torch.Tensor, d: int) -> torch.Tensor:
    """Map skew parameters to orthogonal matrices via the matrix exponential."""
    return torch.matrix_exp(skew_from_params(params, d))


def orthogonality_error(maps: torch.Tensor) -> float:
    """Return max absolute error of ``Q.T @ Q == I`` over a batch of maps."""
    if maps.ndim < 2 or maps.shape[-1] != maps.shape[-2]:
        raise ValueError("maps must have shape (..., d, d)")
    d = maps.shape[-1]
    eye = torch.eye(d, dtype=maps.dtype, device=maps.device)
    gram = maps.transpose(-1, -2) @ maps
    return (gram - eye).abs().max().item()
