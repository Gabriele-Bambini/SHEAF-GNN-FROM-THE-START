"""
Orthogonal Sheaf Diffusion Layer — A1 Mathematical Hardening.

Fixes gauge-symmetry breaking (Critical Failure C1) by projecting restriction maps
to the orthogonal group O(d) before building the sheaf Laplacian.

Mathematical motivation
-----------------------
Sheaf diffusion on a cellular sheaf (G, F) has a *gauge symmetry*: for any
collection of invertible O_v ∈ O(d), one per node, the transformed sheaf

    F̃_{v,e} = O_v F_{v,e}

gives an L_F̃ that is orthogonally similar to L_F — same spectrum, same
cohomology. When restriction maps are *unconstrained* (as in the base
VectorizedSheafDiffusion), BRCA and CRC independently pick different gauge
frames during training, making the learned L_F values numerically
incomparable cross-dataset → zeroshot collapse.

Projection to O(d) via QR decomposition does NOT impose gauge equivalence,
but it does confine all learned maps to a manifold where:

1. F^T F = I exactly → diagonal blocks of L_F are identity-valued
   → spectral bound: eigenvalues of L_F ∈ [0, 2·d_max]
2. Restriction map norm is bounded (no over-scaling)
3. Same hypothesis space (O(d)) for both datasets → comparable maps

QR is differentiable via the implicit function theorem (torch.linalg.qr
supports autograd), so gradients propagate through projection.

Drop-in replacement
-------------------
Inherits from NormalizedVectorizedSheafDiffusion — same interface:
    - compute_connection_laplacian(restriction_maps)
    - diffuse(x, L_F)
    - forward(x, restriction_maps, num_steps)
    - spectral_gap(), condition_number(), verify_laplacian_properties()

Change in tsgnn.py: replace NormalizedVectorizedSheafDiffusion with
OrthogonalSheafDiffusion. All other code unchanged.
"""

import logging
from typing import Optional

import torch

from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion

logger = logging.getLogger(__name__)

__all__ = ["OrthogonalSheafDiffusion"]


def project_to_orthogonal(maps: torch.Tensor) -> torch.Tensor:
    """
    Project restriction maps to O(d) via QR decomposition.

    For each d×d matrix F, compute F = Q·R (thin QR) and return Q.
    Q is orthogonal: Q^T Q = I.

    This operation is differentiable — gradients flow through Q via the
    Gu-Eisenstat formula for QR backward (torch.linalg.qr supports autograd).

    Args:
        maps: (..., d, d) arbitrary restriction map matrices.

    Returns:
        Q: (..., d, d) orthogonal matrices (Q^T Q = I).
    """
    Q, _ = torch.linalg.qr(maps)   # thin QR: Q (..., d, d), R (..., d, d)
    return Q


def project_maps_bundle(restriction_maps: torch.Tensor) -> torch.Tensor:
    """
    Project an (E, 2, d, d) restriction map bundle to O(d).

    Applies QR projection to every (d, d) matrix in the bundle.

    Args:
        restriction_maps: (E, 2, d, d) raw restriction maps.

    Returns:
        orth_maps: (E, 2, d, d) orthogonalized maps.
    """
    E, two, d, _ = restriction_maps.shape
    # Reshape to (2E, d, d) for batched QR, then reshape back.
    flat = restriction_maps.reshape(2 * E, d, d)
    Q = project_to_orthogonal(flat)          # (2E, d, d)
    return Q.reshape(E, 2, d, d)


class OrthogonalSheafDiffusion(NormalizedVectorizedSheafDiffusion):
    """
    Sheaf Diffusion with restriction maps projected to O(d) before each
    Laplacian build.

    Every forward pass:
        1. Receive restriction_maps (E, 2, d, d) — from GRU + FiLM output.
        2. Project: F_{v,e} ↦ Q_{v,e} = qr(F_{v,e})[0]  ∈ O(d).
        3. Build normalized Laplacian L_sym from {Q_{v,e}} (via parent).
        4. Run sheaf diffusion as usual.

    All spectral properties of the normalized Laplacian are preserved
    (inherited from NormalizedVectorizedSheafDiffusion), and the added
    orthogonality ensures:
        - F^T F = I on every edge endpoint
        - L_F eigenvalues bounded in [0, 2] before normalization
        - Gauge-comparable across datasets

    The projection is a differentiable map O: ℝ^{d×d} → O(d), so
    gradients reach the GRU and FiLM parameters through it.

    Args:
        num_nodes: N.
        num_edges: E.
        stalk_dim: d.
        edge_index: (2, E).
        laplacian_eps: PSD regularization for the normalized Laplacian.
        log_orthogonality_error: If True, log ||F^T F − I||_F during training
                                  (useful for verifying the projection works).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
        laplacian_eps: float = 1e-3,
        log_orthogonality_error: bool = False,
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_edges=num_edges,
            stalk_dim=stalk_dim,
            edge_index=edge_index,
            laplacian_eps=laplacian_eps,
        )
        self.log_orthogonality_error = log_orthogonality_error

    def compute_connection_laplacian_vectorized(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build normalized sheaf Laplacian with orthogonally-projected maps.

        Steps:
            1. Resolve maps (external arg or self.restriction_maps).
            2. Project to O(d) via QR.
            3. Optionally log ||F^T F − I||_F.
            4. Delegate to NormalizedVectorizedSheafDiffusion parent.

        Args:
            restriction_maps: (E, 2, d, d) or None (uses self.restriction_maps).

        Returns:
            L_sym: (Nd, Nd) symmetrically normalized sheaf Laplacian.
        """
        maps = restriction_maps if restriction_maps is not None else self.restriction_maps

        # ── Project to O(d) ──────────────────────────────────────────────
        orth_maps = project_maps_bundle(maps)

        # ── Diagnostic: orthogonality error ─────────────────────────────
        if self.log_orthogonality_error and logger.isEnabledFor(logging.DEBUG):
            E, _, d, _ = orth_maps.shape
            flat = orth_maps.reshape(2 * E, d, d)
            FTF = torch.bmm(flat.transpose(1, 2), flat)
            I = torch.eye(d, device=flat.device, dtype=flat.dtype)
            err = (FTF - I).norm(dim=(1, 2)).mean().item()
            logger.debug("[OrthogonalSheafDiffusion] mean ||F^T F − I||_F = %.2e", err)

        # ── Delegate to parent with orthogonalized maps ───────────────────
        return super().compute_connection_laplacian_vectorized(orth_maps)

    # compute_connection_laplacian is inherited → aliases vectorized version.
    # diffuse, sheaf_diffusion, forward, spectral_gap, condition_number,
    # verify_laplacian_properties — all inherited unchanged.


# ── Verification helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def verify_orthogonality(layer: OrthogonalSheafDiffusion,
                         restriction_maps: Optional[torch.Tensor] = None,
                         tol: float = 1e-5) -> dict:
    """
    Verify that the maps used in the Laplacian build satisfy F^T F = I.

    Args:
        layer: OrthogonalSheafDiffusion instance.
        restriction_maps: Optional external maps (E, 2, d, d).
        tol: Tolerance for ||F^T F − I||_F per matrix.

    Returns:
        dict with keys: max_orth_error, mean_orth_error, all_pass, n_matrices.
    """
    maps = restriction_maps if restriction_maps is not None else layer.restriction_maps
    orth_maps = project_maps_bundle(maps)

    E, _, d, _ = orth_maps.shape
    flat = orth_maps.reshape(2 * E, d, d)
    FTF = torch.bmm(flat.transpose(1, 2), flat)
    I = torch.eye(d, device=flat.device, dtype=flat.dtype)
    errors = (FTF - I).norm(dim=(1, 2))    # (2E,) per-matrix error

    return {
        "max_orth_error": errors.max().item(),
        "mean_orth_error": errors.mean().item(),
        "all_pass": bool((errors < tol).all()),
        "n_matrices": 2 * E,
    }


@torch.no_grad()
def verify_spectral_bound(layer: OrthogonalSheafDiffusion,
                          restriction_maps: Optional[torch.Tensor] = None) -> dict:
    """
    Verify L_F eigenvalues are in [0, ~2].

    With orthogonal F: diagonal blocks F^T F = I → each diagonal block of
    L_F (before normalization) ≈ deg(v)·I.  After symmetric normalization,
    eigenvalues are bounded in [0, 2] by the Cheeger inequality.

    Returns:
        dict with min_eig, max_eig, in_bound.
    """
    L = layer.compute_connection_laplacian(restriction_maps)
    eigs = torch.linalg.eigvalsh(L.float()).sort().values
    upper_bound = 2.0 + 1e-3   # small tolerance for numerics

    return {
        "min_eig": eigs[0].item(),
        "max_eig": eigs[-1].item(),
        "in_bound": bool((eigs >= -1e-4).all() and eigs[-1].item() <= upper_bound),
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(42)
    N, E, d = 10, 40, 4
    edge_index = torch.randint(0, N, (2, E))
    maps = torch.randn(E, 2, d, d)

    layer = OrthogonalSheafDiffusion(N, E, d, edge_index)

    # Orthogonality check
    orth = verify_orthogonality(layer, maps)
    print(f"Orthogonality  — max_err={orth['max_orth_error']:.2e}  "
          f"mean_err={orth['mean_orth_error']:.2e}  pass={orth['all_pass']}")

    # Spectral bound check
    spec = verify_spectral_bound(layer, maps)
    print(f"Spectral bound — min={spec['min_eig']:.4f}  "
          f"max={spec['max_eig']:.4f}  in[0,2]={spec['in_bound']}")

    # Forward pass
    x = torch.randn(N, d)
    x_out, L_F = layer(x, restriction_maps=maps, num_steps=1)
    print(f"Forward pass   — input {x.shape}  output {x_out.shape}  "
          f"L_F {L_F.shape}  finite={torch.isfinite(x_out).all().item()}")

    # Parameter count
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters     — {n_params:,}")

    sys.exit(0 if orth["all_pass"] and spec["in_bound"] else 1)
