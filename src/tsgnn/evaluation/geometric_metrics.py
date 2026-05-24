"""
Post-hoc geometric metrics for the Lie-Sheaf allele model.

These are computed AFTER training on the trained model's restriction maps.
They are the PRIMARY biological readout -- expression reconstruction is secondary.

Metrics:
    edge_restriction_shift:  ||F_allele[e] - F_WT[e]||_F per edge
    so_distance_matrix:      pairwise SO(d) distances between allele class rotations
    h1_dimension_approx:     approximate dim of H1 sheaf cohomology (null space of L_F)
    allele_divergence:       ||pred_allele - pred_WT||_2 (training diagnostic)
"""

import torch

__all__ = [
    "edge_restriction_shift",
    "so_distance_matrix",
    "h1_dimension_approx",
    "allele_divergence",
]


def edge_restriction_shift(
    F_allele: torch.Tensor,
    F_wt: torch.Tensor,
) -> torch.Tensor:
    """
    Per-edge Frobenius norm of restriction map change vs WT.

    Args:
        F_allele: (E, 2, d, d) restriction maps for a given allele class.
        F_wt:     (E, 2, d, d) restriction maps for WT.

    Returns:
        shifts: (E,) -- how much each edge's restriction map changed.
                High values = edge strongly affected by this allele class.
    """
    diff = F_allele - F_wt  # (E, 2, d, d)
    # Mean over the two endpoints (source + target), Frobenius norm over (d, d)
    return diff.norm(dim=(-2, -1)).mean(dim=1)  # (E,)


def so_distance_matrix(R_matrices: torch.Tensor) -> torch.Tensor:
    """
    Pairwise SO(d) distances between rotation matrices.

    Distance: ||R_i - R_j||_F / sqrt(2*d) in [0, 1]

    Args:
        R_matrices: (n, d, d) tensor of rotation matrices.

    Returns:
        D: (n, n) symmetric matrix of pairwise distances.
    """
    n, d, _ = R_matrices.shape
    # Max ||R_i - R_j||_F for R in SO(d) is 2*sqrt(d) (e.g. I vs -I for even d)
    norm_factor = 2.0 * (d ** 0.5)
    D = torch.zeros(n, n, device=R_matrices.device)
    for i in range(n):
        for j in range(n):
            D[i, j] = (R_matrices[i] - R_matrices[j]).norm() / norm_factor
    return D


def h1_dimension_approx(L_F: torch.Tensor, eps: float = 1e-4) -> int:
    """
    Approximate dimension of H1 sheaf cohomology.

    For a cellular sheaf on a graph, H1 = coker(delta_0) ≈ nullity of L_F.
    Concretely: number of eigenvalues of L_F below `eps`.

    Args:
        L_F: (N*d, N*d) sheaf connection Laplacian (symmetric PSD).
        eps: threshold for "zero" eigenvalue (default 1e-4).

    Returns:
        dim_h1: integer -- dimension of H1.
                0 = globally consistent sheaf (no cohomological defects).
                Large = many inconsistencies in the regulatory network.
    """
    eigenvalues = torch.linalg.eigvalsh(L_F)  # ascending order
    return int((eigenvalues < eps).sum().item())


def allele_divergence(pred_allele: torch.Tensor, pred_wt: torch.Tensor) -> float:
    """
    Mean L2 distance between allele and WT predictions over all genes.

    Used as training diagnostic -- should be > 0.01 for the model to
    differentiate allele classes. If < 0.001, the SO(4) embedding is not
    producing distinguishable outputs (rotation collapsed to identity).

    Args:
        pred_allele: (N,) or (K, N) predicted expression for allele class.
        pred_wt:     (N,) or (K, N) predicted expression for WT.

    Returns:
        divergence: float scalar.
    """
    return (pred_allele - pred_wt).norm(dim=-1).mean().item()
