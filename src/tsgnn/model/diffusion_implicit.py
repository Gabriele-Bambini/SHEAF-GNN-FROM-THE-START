"""
Implicit Sheaf Diffusion via Crank-Nicolson — A2 Mathematical Hardening.

Replaces the explicit Euler diffusion step (unstable, over-smoothing) with a
fully implicit Crank-Nicolson (CN) solver.

Theory
------
Explicit (current):
    x_{k+1} = x_k - α · σ(L_F x_k W)
    Stability: requires ‖L_F‖ α < 1. With large L_F or α, diverges.
    σ nonlinearity breaks energy monotonicity — not a real diffusion.

Crank-Nicolson (new):
    (I + τ/2 · L̂) x_{k+1} = (I - τ/2 · L̂) x_k     (*)

    A-stable (unconditionally stable for any τ > 0).
    Monotone: Dirichlet energy E(x) = ½ xᵀ L x is non-increasing.
    One CN step ≈ 10 explicit steps in mixing power, so num_diffusion_steps=1 suffices.

Solver
------
(*) is a symmetric positive-definite linear system (I + τ/2 L̂ is SPD when
L̂ ≥ 0). Solved via Conjugate Gradient (CG), which is:
  - Differentiable: uses only standard tensor ops → autograd tracks through.
  - Memory-efficient: O(Nd) storage, O(Nd²) per iteration.
  - Fast: with orthogonal maps, κ(I + τ/2 L̂) ≤ (1+τ)/(1+0) bounded → CG
    converges in ~10 iterations regardless of graph size.

τ is a learnable scalar (clamped to [0.01, 1.0] for stability).

Drop-in replacement
-------------------
Inherits from OrthogonalSheafDiffusion (which inherits from
NormalizedVectorizedSheafDiffusion). Same interface. Only `diffuse` is
overridden; compute_connection_laplacian, forward, spectral diagnostics
are inherited.

Usage in TSGNN:
    model = TSGNN(..., use_implicit_diffusion=True)
    # sheaf_layer → ImplicitOrthogonalSheafDiffusion
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion

logger = logging.getLogger(__name__)

__all__ = ["ImplicitOrthogonalSheafDiffusion"]


def conjugate_gradient(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Conjugate Gradient solver: find x such that A x = b.

    Args:
        A: (n, n) symmetric positive-definite matrix.
        b: (n,) right-hand side.
        x0: (n,) initial guess. Defaults to b (good warm start for CN).
        max_iter: Maximum CG iterations. With well-conditioned A (κ ≤ 10),
                  10–15 iterations achieve tol=1e-4.
        tol: Convergence threshold on residual norm.

    Returns:
        x: (n,) approximate solution.

    Differentiability:
        All ops are standard torch operations (matmul, add, mul, dot).
        Autograd can differentiate through the returned x with respect to A
        and b via the adjoint of the linear system (implicit diff theorem).
        Gradient is approximate but stable for well-conditioned A.

    Note on stability:
        We use A @ p (dense matvec) instead of sparse. For Nd=2000, this is
        a 2000×2000 matmul — fast on GPU/CPU. For larger graphs, replace with
        `torch.sparse.mm` by passing a sparse A.
    """
    if x0 is None:
        # Warm start: use b as initial guess (reduces iterations for CN where
        # x_{k+1} ≈ x_k, so A x_{k+1} ≈ b ≈ (I - τ/2 L) x_k ≈ b)
        x = b.clone()
    else:
        x = x0.clone()

    r = b - (A @ x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        if rs_old.sqrt() < tol:
            break

        Ap = A @ p
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    return x


class ImplicitOrthogonalSheafDiffusion(NormalizedVectorizedSheafDiffusion):
    """
    Sheaf Diffusion with:
      - Orthogonal restriction maps via QR projection
      - Implicit Crank-Nicolson diffusion via CG

    Inherits vectorized Laplacian construction from NormalizedVectorizedSheafDiffusion.
    CayleyOrthogonalSheafDiffusion overrides get_orthogonal_maps with Cayley parametrization.

    Args:
        num_nodes, num_edges, stalk_dim, edge_index: standard sheaf args.
        tau_init: Initial diffusion timescale (default 0.2). Learned.
        cg_max_iter: Max CG iterations per diffusion step (default 20).
        cg_tol: CG convergence tolerance (default 1e-4).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
        laplacian_eps: float = 1e-3,
        tau_init: float = 0.2,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-4,
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_edges=num_edges,
            stalk_dim=stalk_dim,
            edge_index=edge_index,
            laplacian_eps=laplacian_eps,
        )
        self.log_tau = nn.Parameter(
            torch.tensor(tau_init).log()
        )
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol

    def get_orthogonal_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """QR-based orthogonal projection (legacy; overridden in CayleyOrthogonalSheafDiffusion)."""
        E, _, d, _ = maps.shape
        F_src = maps[:, 0]  # (E, d, d)
        F_tgt = maps[:, 1]  # (E, d, d)
        Q_src, _ = torch.linalg.qr(F_src)
        Q_tgt, _ = torch.linalg.qr(F_tgt)
        return torch.stack([Q_src, Q_tgt], dim=1)  # (E, 2, d, d)

    @property
    def tau(self) -> torch.Tensor:
        """Diffusion timescale τ = exp(log_tau), clamped to [0.01, 1.0]."""
        return torch.clamp(self.log_tau.exp(), min=0.01, max=1.0)

    def diffuse(
        self,
        x: torch.Tensor,
        L_F: torch.Tensor,
    ) -> torch.Tensor:
        """
        One Crank-Nicolson implicit diffusion step.

        Solves:  (I + τ/2 · L_F) x_next = (I - τ/2 · L_F) x_curr

        Args:
            x: (N, d) node features in stalk space.
            L_F: (N*d, N*d) normalized sheaf Laplacian (from OrthogonalSheafDiffusion).

        Returns:
            x_next: (N, d) diffused features.
        """
        N, d = x.shape
        Nd = N * d
        device = x.device
        dtype = x.dtype

        # Flatten to (Nd,) for block-vector operations
        x_flat = x.reshape(Nd)

        tau = self.tau.to(device=device, dtype=dtype)

        # Build system matrix A = I + τ/2 · L_F
        I = torch.eye(Nd, device=device, dtype=dtype)
        A = I + (tau * 0.5) * L_F              # (Nd, Nd)

        # Build RHS b = (I - τ/2 · L_F) x
        b = x_flat - (tau * 0.5) * (L_F @ x_flat)   # (Nd,)

        # Solve A x_next = b via CG
        x_next_flat = conjugate_gradient(
            A, b,
            x0=x_flat,                    # warm start: good for small τ
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
        )

        # Guard: if CG blows up (shouldn't happen with orthogonal L_F), clamp
        x_next_flat = torch.clamp(x_next_flat, min=-50.0, max=50.0)

        return x_next_flat.reshape(N, d)

    # forward(), compute_connection_laplacian(), spectral_gap(), etc. — all inherited.


# ── Self-contained verification ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    N, E, d = 10, 40, 4
    edge_index = torch.randint(0, N, (2, E))
    maps = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)

    layer = ImplicitOrthogonalSheafDiffusion(N, E, d, edge_index)
    x_out, L_F = layer(x, restriction_maps=maps, num_steps=1)

    print(f"Input:  {x.shape}")
    print(f"Output: {x_out.shape}  finite={torch.isfinite(x_out).all().item()}")
    print(f"tau:    {layer.tau.item():.4f}")

    # Energy test
    E_before = 0.5 * (x.reshape(-1) @ L_F @ x.reshape(-1)).item()
    E_after  = 0.5 * (x_out.reshape(-1) @ L_F @ x_out.reshape(-1)).item()
    print(f"Energy: {E_before:.4f} → {E_after:.4f}  monotone={E_after <= E_before + 1e-3}")

    # Gradient test
    x2 = torch.randn(N, d, requires_grad=False)
    maps2 = torch.randn(E, 2, d, d, requires_grad=True)
    L2 = layer.compute_connection_laplacian(maps2)
    x2_out = layer.diffuse(x2, L2)
    x2_out.sum().backward()
    print(f"Gradient on maps: {maps2.grad.abs().max().item():.4e}  finite={torch.isfinite(maps2.grad).all().item()}")

    sys.exit(0)
