"""Coboundary construction for finite graph cellular sheaves."""

from __future__ import annotations

import torch

__all__ = ["build_dense_coboundary"]


def _validate_inputs(
    num_nodes: int,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
) -> tuple[int, int]:
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if maps.ndim != 4 or maps.shape[1] != 2 or maps.shape[-1] != maps.shape[-2]:
        raise ValueError("maps must have shape (E, 2, d, d)")
    if edge_index.shape[1] != maps.shape[0]:
        raise ValueError(
            f"edge count mismatch: edge_index has {edge_index.shape[1]}, "
            f"maps has {maps.shape[0]}"
        )
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
    ):
        raise ValueError("edge_index contains node ids outside [0, num_nodes)")
    return maps.shape[0], maps.shape[-1]


def _coerce_edge_weights(
    edge_weights: torch.Tensor | None,
    num_edges: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if edge_weights is None:
        return torch.ones(num_edges, dtype=dtype, device=device)
    weights = edge_weights.to(dtype=dtype, device=device).reshape(-1)
    if weights.numel() != num_edges:
        raise ValueError(f"edge_weights must have {num_edges} entries")
    if (weights < 0).any():
        raise ValueError("edge_weights must be non-negative")
    return weights


def build_dense_coboundary(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    maps: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build dense coboundary ``B`` so the sheaf Laplacian is ``B.T @ B``.

    For edge ``e = (u, v)``, the row block is
    ``[-sqrt(w_e) F_{u<e}, +sqrt(w_e) F_{v<e}]``.
    """
    num_edges, d = _validate_inputs(num_nodes, edge_index, maps)
    weights = _coerce_edge_weights(
        edge_weights,
        num_edges,
        dtype=maps.dtype,
        device=maps.device,
    )
    edge_index = edge_index.to(device=maps.device, dtype=torch.long)

    coboundary = torch.zeros(
        num_edges * d,
        num_nodes * d,
        dtype=maps.dtype,
        device=maps.device,
    )
    sqrt_weights = weights.sqrt()

    for edge_id in range(num_edges):
        src = int(edge_index[0, edge_id].item())
        tgt = int(edge_index[1, edge_id].item())
        row = slice(edge_id * d, (edge_id + 1) * d)
        src_col = slice(src * d, (src + 1) * d)
        tgt_col = slice(tgt * d, (tgt + 1) * d)
        scale = sqrt_weights[edge_id]
        coboundary[row, src_col] = -scale * maps[edge_id, 0]
        coboundary[row, tgt_col] = scale * maps[edge_id, 1]

    return coboundary
