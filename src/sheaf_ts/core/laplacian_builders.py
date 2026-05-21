"""Dense sheaf Laplacian builders with explicit mathematical contracts."""

from __future__ import annotations

import torch

from sheaf_ts.core.coboundary import _coerce_edge_weights, _validate_inputs
from sheaf_ts.core.orthogonal import orthogonality_error

__all__ = [
    "build_connection_laplacian",
    "build_general_laplacian",
    "normalize_block_laplacian",
]


def _empty_laplacian(num_nodes: int, d: int, maps: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        num_nodes * num_nodes,
        d,
        d,
        dtype=maps.dtype,
        device=maps.device,
    )


def _block_laplacian_to_dense(blocks: torch.Tensor, num_nodes: int, d: int) -> torch.Tensor:
    return blocks.view(num_nodes, num_nodes, d, d).permute(0, 2, 1, 3).reshape(
        num_nodes * d,
        num_nodes * d,
    )


def _add_regularizer(laplacian: torch.Tensor, eps: float) -> torch.Tensor:
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if eps == 0:
        return laplacian
    eye = torch.eye(laplacian.shape[0], dtype=laplacian.dtype, device=laplacian.device)
    return laplacian + eps * eye


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def _inv_sqrt_psd(block: torch.Tensor, *, atol: float = 1e-10) -> torch.Tensor:
    block = _symmetrize(block)
    eigenvalues, eigenvectors = torch.linalg.eigh(block)
    min_eigenvalue = eigenvalues.min().item()
    if min_eigenvalue < -atol:
        raise ValueError(f"degree block is not PSD: min eigenvalue {min_eigenvalue:.3e}")
    inv = torch.zeros_like(eigenvalues)
    mask = eigenvalues > atol
    inv[mask] = eigenvalues[mask].rsqrt()
    return (eigenvectors * inv.unsqueeze(0)) @ eigenvectors.T


def normalize_block_laplacian(
    laplacian: torch.Tensor,
    *,
    num_nodes: int,
    stalk_dim: int,
    degree_regularizer: float = 0.0,
) -> torch.Tensor:
    """Return ``D^{-1/2} L D^{-1/2}`` using diagonal ``d x d`` blocks of ``L``."""
    expected = num_nodes * stalk_dim
    if laplacian.shape != (expected, expected):
        raise ValueError(
            f"laplacian shape {tuple(laplacian.shape)} does not match "
            f"num_nodes * stalk_dim = {expected}"
        )
    if degree_regularizer < 0:
        raise ValueError("degree_regularizer must be non-negative")

    normalizer = torch.zeros_like(laplacian)
    eye = torch.eye(stalk_dim, dtype=laplacian.dtype, device=laplacian.device)

    for node in range(num_nodes):
        slc = slice(node * stalk_dim, (node + 1) * stalk_dim)
        degree_block = laplacian[slc, slc]
        if degree_regularizer:
            degree_block = degree_block + degree_regularizer * eye
        normalizer[slc, slc] = _inv_sqrt_psd(degree_block)

    return _symmetrize(normalizer @ laplacian @ normalizer)


def _prepare(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    edge_weights: torch.Tensor | None,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    num_edges, d = _validate_inputs(num_nodes, edge_index, maps)
    edge_index = edge_index.to(device=maps.device, dtype=torch.long)
    weights = _coerce_edge_weights(
        edge_weights,
        num_edges,
        dtype=maps.dtype,
        device=maps.device,
    )
    return num_edges, d, edge_index, weights


def build_general_laplacian(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
    normalize: bool = False,
    eps: float = 0.0,
    degree_regularizer: float = 0.0,
) -> torch.Tensor:
    """Build a sheaf Laplacian for arbitrary restriction maps.

    This is the safe default. The diagonal is always computed as
    ``sum_e F_{v<e}.T @ F_{v<e}``, so non-orthogonal maps remain valid.
    """
    num_edges, d, edge_index, weights = _prepare(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
    )
    src = edge_index[0]
    tgt = edge_index[1]
    left = maps[:, 0]
    right = maps[:, 1]
    weight_blocks = weights.view(num_edges, 1, 1)

    diag_src = weight_blocks * torch.bmm(left.transpose(1, 2), left)
    diag_tgt = weight_blocks * torch.bmm(right.transpose(1, 2), right)
    off_src_tgt = -weight_blocks * torch.bmm(left.transpose(1, 2), right)
    off_tgt_src = off_src_tgt.transpose(1, 2)

    blocks = _empty_laplacian(num_nodes, d, maps)
    blocks.index_add_(0, src * num_nodes + src, diag_src)
    blocks.index_add_(0, tgt * num_nodes + tgt, diag_tgt)
    blocks.index_add_(0, src * num_nodes + tgt, off_src_tgt)
    blocks.index_add_(0, tgt * num_nodes + src, off_tgt_src)

    laplacian = _symmetrize(_block_laplacian_to_dense(blocks, num_nodes, d))
    laplacian = _add_regularizer(laplacian, eps)
    if normalize:
        laplacian = normalize_block_laplacian(
            laplacian,
            num_nodes=num_nodes,
            stalk_dim=d,
            degree_regularizer=degree_regularizer,
        )
    return laplacian


def build_connection_laplacian(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 0.0,
    degree_regularizer: float = 0.0,
    verify_orthogonal: bool = True,
    orthogonal_atol: float = 1e-6,
) -> torch.Tensor:
    """Build a normalized connection Laplacian for orthogonal maps.

    This builder may use the faster diagonal ``sum w_e * I`` only because
    orthogonality implies ``Q.T @ Q = I``. When ``verify_orthogonal`` is true,
    non-orthogonal maps fail closed instead of silently producing a fake
    sheaf Laplacian.
    """
    num_edges, d, edge_index, weights = _prepare(
        num_nodes=num_nodes,
        edge_index=edge_index,
        maps=maps,
        edge_weights=edge_weights,
    )
    if verify_orthogonal:
        err = orthogonality_error(maps)
        if err > orthogonal_atol:
            raise ValueError(
                "connection Laplacian requires orthogonal maps; "
                f"max Q.T@Q-I error is {err:.3e}"
            )

    src = edge_index[0]
    tgt = edge_index[1]
    left = maps[:, 0]
    right = maps[:, 1]
    weight_blocks = weights.view(num_edges, 1, 1)
    eye = torch.eye(d, dtype=maps.dtype, device=maps.device).expand(num_edges, d, d)

    diag_src = weight_blocks * eye
    diag_tgt = weight_blocks * eye
    off_src_tgt = -weight_blocks * torch.bmm(left.transpose(1, 2), right)
    off_tgt_src = off_src_tgt.transpose(1, 2)

    blocks = _empty_laplacian(num_nodes, d, maps)
    blocks.index_add_(0, src * num_nodes + src, diag_src)
    blocks.index_add_(0, tgt * num_nodes + tgt, diag_tgt)
    blocks.index_add_(0, src * num_nodes + tgt, off_src_tgt)
    blocks.index_add_(0, tgt * num_nodes + src, off_tgt_src)

    laplacian = _symmetrize(_block_laplacian_to_dense(blocks, num_nodes, d))
    laplacian = _add_regularizer(laplacian, eps)
    if normalize:
        laplacian = normalize_block_laplacian(
            laplacian,
            num_nodes=num_nodes,
            stalk_dim=d,
            degree_regularizer=degree_regularizer,
        )
    return laplacian
