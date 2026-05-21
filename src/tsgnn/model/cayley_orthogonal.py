"""
TSGNN-v2 Step 1: Direct O(d) Parametrization via Cayley Transform.

Problem fixed
-------------
The old pipeline produced raw restriction maps (E, 2, d, d) and then
projected them to O(d) via QR at Laplacian build time.  QR kills an entire
right-triangular orbit — d(d+1)/2 dimensions per map.  With d=6 that is
21 out of 36 parameters, making any FiLM conditioning applied *before* QR
almost completely non-identifiable.

Solution
--------
Parametrize the orthogonal group *directly* using skew-symmetric matrices:

    A_e  ∈  so(d)   —  d(d-1)/2 free parameters per endpoint
    Q_e  =  cayley(A_e)  =  (I - A_e)(I + A_e)^{-1}  ∈  SO(d)

Benefits
~~~~~~~~
* No post-hoc projection → no nullspace.
* Exactly d(d-1)/2 free parameters — the dimension of O(d).
* Globally smooth, except at A such that -1 is an eigenvalue of A
  (measure-zero set — irrelevant in practice for small d).
* Differentiable everywhere else → autograd works out of the box.
* Cheaper than matrix exponential for d ≤ 8 (only one linear solve).

Gradient check
~~~~~~~~~~~~~~
torch.linalg.solve is differentiable w.r.t. both arguments since PyTorch
≥ 1.9.  The Jacobian of (I+A)^{-1} w.r.t. A is bounded whenever (I+A)
is well-conditioned, which holds whenever ||A|| < 1 — satisfied during
normal training with small-weight initialization.

Usage
-----
    from tsgnn.model.cayley_orthogonal import (
        skew_to_orthogonal,
        SkewParametrizedMaps,
        CayleyOrthogonalSheafDiffusion,
    )

    # Low-level
    A = torch.randn(E, 2, d*(d-1)//2)       # free parameters
    Q = skew_to_orthogonal(A, d)              # (E, 2, d, d)

    # Module (drop-in for the old OrthogonalSheafDiffusion)
    layer = CayleyOrthogonalSheafDiffusion(N, E, d, edge_index)
    L_F   = layer.compute_connection_laplacian()   # uses Cayley maps
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .diffusion_implicit import ImplicitOrthogonalSheafDiffusion

logger = logging.getLogger(__name__)

__all__ = [
    "skew_from_vector",
    "skew_to_orthogonal",
    "SkewParametrizedMaps",
    "CayleyOrthogonalSheafDiffusion",
    "CayleyOrthogonalSheafDiffusionExplicit",
]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def skew_from_vector(v: torch.Tensor, d: int) -> torch.Tensor:
    """
    Build a batch of d×d skew-symmetric matrices from free-parameter vectors.

    Args:
        v: (..., d*(d-1)//2) packed upper-triangle parameters.
        d: Stalk dimension.

    Returns:
        A: (..., d, d) skew-symmetric matrices (A^T = -A, diagonal=0).
    """
    k = d * (d - 1) // 2
    assert v.shape[-1] == k, f"Expected {k} params for d={d}, got {v.shape[-1]}"

    batch_shape = v.shape[:-1]
    A = torch.zeros(*batch_shape, d, d, device=v.device, dtype=v.dtype)

    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[..., i, j] = v[..., idx]
            A[..., j, i] = -v[..., idx]
            idx += 1

    return A


def skew_to_orthogonal(v: torch.Tensor, d: int) -> torch.Tensor:
    """
    Map free parameters → SO(d) via Cayley transform.

    Cayley:  Q = (I - A)(I + A)^{-1}   where  A = skew(v).

    Args:
        v: (..., d*(d-1)//2) free parameters.
        d: Stalk dimension.

    Returns:
        Q: (..., d, d) orthogonal matrices (Q^T Q = I, det Q = +1 near init).
    """
    A = skew_from_vector(v, d)                       # (..., d, d)
    batch_shape = v.shape[:-1]

    I = torch.eye(d, device=v.device, dtype=v.dtype).expand(*batch_shape, d, d)

    ImA = I - A   # (I - A)
    IpA = I + A   # (I + A)

    # Q = (I - A) @ (I + A)^{-1}
    # torch.linalg.solve(B, C) computes B^{-1} C, so:
    #   Q^T = (I + A)^{-1} (I - A) = solve(I+A, I-A)
    # Then Q = Q^T^T = (solve(I+A, I-A))^T
    # Damping: add small epsilon to diagonal of (I+A) for numerical stability.
    eps = 1e-6
    IpA_reg = IpA + eps * I
    Qt = torch.linalg.solve(IpA_reg, ImA)               # (..., d, d)
    Q  = Qt.transpose(-2, -1)
    return Q


# ── Learnable skew-parametrized restriction maps ──────────────────────────────

class SkewParametrizedMaps(nn.Module):
    """
    Learnable restriction maps parametrized as SO(d) via Cayley transform.

    Each edge endpoint gets d(d-1)//2 free parameters (a skew-symmetric
    matrix), which is mapped to O(d) on every forward call.

    No QR projection needed → all parameters are identifiable.

    Args:
        num_edges:  E, number of directed edges.
        stalk_dim:  d.
        scale_init: Std of Gaussian initialization for the free parameters.
                    Small (0.1) → Q ≈ I at start (stable diffusion).
    """

    def __init__(self, num_edges: int, stalk_dim: int, scale_init: float = 0.1):
        super().__init__()
        self.E = num_edges
        self.d = stalk_dim
        self.k = stalk_dim * (stalk_dim - 1) // 2  # free params per (edge, endpoint)

        # (E, 2, k)  —  2 endpoints per edge
        self.skew_params = nn.Parameter(
            torch.randn(num_edges, 2, self.k) * scale_init
        )

    def forward(self) -> torch.Tensor:
        """
        Returns:
            Q: (E, 2, d, d) orthogonal restriction maps.
        """
        # skew_params: (E, 2, k)
        # Process both endpoints together by merging batch dims
        flat = self.skew_params.reshape(self.E * 2, self.k)   # (2E, k)
        Q_flat = skew_to_orthogonal(flat, self.d)              # (2E, d, d)
        return Q_flat.reshape(self.E, 2, self.d, self.d)       # (E, 2, d, d)

    def extra_repr(self) -> str:
        return (
            f"num_edges={self.E}, stalk_dim={self.d}, "
            f"free_params_per_map={self.k}, total_params={self.skew_params.numel()}"
        )


# ── Full sheaf diffusion layer with Cayley maps ───────────────────────────────

class CayleyOrthogonalSheafDiffusion(ImplicitOrthogonalSheafDiffusion):
    """
    The compute_connection_laplacian() method is overridden to:
    1. Accept an optional (E, 2, d, d) argument of *already-orthogonal* maps.
    2. When called without argument, produce maps from SkewParametrizedMaps.
    3. Skip the QR projection step entirely.

    Supports both implicit (Crank-Nicolson) and explicit (Forward Euler) diffusion.
    If use_implicit=False, it uses the explicit diffuse() from NormalizedVectorizedSheafDiffusion.
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
        scale_init: float = 0.1,
        use_implicit: bool = True,
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_edges=num_edges,
            stalk_dim=stalk_dim,
            edge_index=edge_index,
            laplacian_eps=laplacian_eps,
            tau_init=tau_init,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )
        self.use_implicit = use_implicit
        # Replace the raw restriction_maps parameter with Cayley parametrization
        # (the parent class also has self.restriction_maps — we shadow it with a module)
        self.cayley_maps = SkewParametrizedMaps(num_edges, stalk_dim, scale_init)

    def diffuse(
        self,
        x: torch.Tensor,
        L_F: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one step of sheaf diffusion.
        If self.use_implicit is True, uses Crank-Nicolson (Implicit).
        Otherwise, uses Forward Euler (Explicit).
        """
        if self.use_implicit:
            # Call ImplicitOrthogonalSheafDiffusion.diffuse
            return super().diffuse(x, L_F)
        else:
            # Fallback to NormalizedVectorizedSheafDiffusion.sheaf_diffusion (Explicit)
            # We call it via the grandparent to skip the solver logic.
            from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion
            return NormalizedVectorizedSheafDiffusion.sheaf_diffusion(self, x, L_F)

    def get_orthogonal_maps(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
        *,
        allow_qr_fallback: bool = False,
    ) -> torch.Tensor:
        """
        Resolve (E, 2, d, d) orthogonal maps.

        Priority:
          1. If restriction_maps is provided (from ODE+FiLM output), use it
             directly — caller is responsible for ensuring orthogonality OR
             we treat it as skew parameters → Cayley.
          2. Otherwise, use self.cayley_maps.

        For TSGNN-v2, the ODE/FiLM output is a (E, 2*k) vector of *skew
        parameters* (not raw d×d matrices), so we apply Cayley here.
        """
        if restriction_maps is None:
            return self.cayley_maps()  # (E, 2, d, d)

        # If caller passes (E, 2, d, d) — ODE output, NOT guaranteed orthogonal.
        # Apply QR to project back to O(d). This keeps orthogonality maintained
        # throughout the ODE trajectory without requiring the ODE to operate in
        # so(d) space (which would require a bigger architectural change).
        if restriction_maps.shape[-1] == self.d and restriction_maps.shape[-2] == self.d:
            if not allow_qr_fallback:
                raise ValueError(
                    "raw matrix restriction maps are not accepted in the Cayley v3 path; "
                    "pass packed skew parameters with shape (E, 2, k) or (E, 2*k)"
                )
            flat = restriction_maps.reshape(-1, self.d, self.d)   # (E*2, d, d)
            Q, _ = torch.linalg.qr(flat)                          # (E*2, d, d)
            return Q.reshape(restriction_maps.shape)               # (E, 2, d, d)

        # If caller passes (E, k_total) flat skew params → Cayley
        E = restriction_maps.shape[0]
        k = self.d * (self.d - 1) // 2
        if restriction_maps.shape[-2:] == (2, k):
            flat = restriction_maps.reshape(E * 2, k)
            Q = skew_to_orthogonal(flat, self.d)
            return Q.reshape(E, 2, self.d, self.d)

        if restriction_maps.shape[-1] == 2 * k:
            flat = restriction_maps.reshape(E * 2, k)
            Q = skew_to_orthogonal(flat, self.d)
            return Q.reshape(E, 2, self.d, self.d)

        # Fallback: unexpected shape — log and apply QR anyway
        raise ValueError(
            "[CayleyOrthogonalSheafDiffusion] unexpected restriction_maps shape "
            f"{tuple(restriction_maps.shape)}; expected (E, 2, k) or (E, 2*k)"
        )

    def compute_connection_laplacian(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the (Nd, Nd) sheaf Laplacian without QR projection.

        Uses get_orthogonal_maps() → delegates to parent's vectorized builder.
        The parent (NormalizedVectorizedSheafDiffusion) accepts the already-
        orthogonal maps and just builds the Laplacian — no internal projection.
        """
        orth_maps = self.get_orthogonal_maps(restriction_maps)
        # Call the grandparent's vectorized builder (skipping OrthogonalSheafDiffusion's
        # QR step by going directly to NormalizedVectorizedSheafDiffusion)
        from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion
        return NormalizedVectorizedSheafDiffusion.compute_connection_laplacian_vectorized(
            self, orth_maps
        )

    # forward(), tau — all inherited from ImplicitOrthogonalSheafDiffusion


class CayleyOrthogonalSheafDiffusionExplicit(CayleyOrthogonalSheafDiffusion):
    """Convenience class for explicit Cayley diffusion."""
    def __init__(self, *args, **kwargs):
        kwargs["use_implicit"] = False
        super().__init__(*args, **kwargs)


# ── Verification ──────────────────────────────────────────────────────────────

@torch.no_grad()
def verify_cayley_orthogonality(d: int = 4, n_samples: int = 100, tol: float = 1e-5) -> dict:
    """Check that Cayley-mapped matrices satisfy Q^T Q = I."""
    k = d * (d - 1) // 2
    v = torch.randn(n_samples, k)
    Q = skew_to_orthogonal(v, d)           # (n_samples, d, d)
    QTQ = torch.bmm(Q.transpose(1, 2), Q)  # (n_samples, d, d)
    I   = torch.eye(d).unsqueeze(0).expand(n_samples, d, d)
    err = (QTQ - I).norm(dim=(1, 2))       # (n_samples,)
    return {
        "max_orth_error":  err.max().item(),
        "mean_orth_error": err.mean().item(),
        "all_pass":        bool((err < tol).all()),
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    for d in [4, 6, 8]:
        result = verify_cayley_orthogonality(d=d)
        status = "PASS" if result["all_pass"] else "FAIL"
        print(
            f"d={d}  [{status}]  max_err={result['max_orth_error']:.2e}  "
            f"mean_err={result['mean_orth_error']:.2e}"
        )

    # Full layer test
    N, E, d = 20, 80, 6
    edge_index = torch.randint(0, N, (2, E))
    layer = CayleyOrthogonalSheafDiffusion(N, E, d, edge_index)

    x = torch.randn(N, d)
    x_out, L_F = layer(x, num_steps=1)
    print(f"\nForward: {x.shape} → {x_out.shape}  finite={torch.isfinite(x_out).all().item()}")
    print(f"L_F: {L_F.shape}  symmetric={torch.allclose(L_F, L_F.T, atol=1e-5)}")

    # Gradient flows through Cayley maps
    v_test = torch.randn(E, 2, d * (d - 1) // 2, requires_grad=True)
    flat   = v_test.reshape(E * 2, d * (d - 1) // 2)
    Q      = skew_to_orthogonal(flat, d)
    loss   = Q.sum()
    loss.backward()
    print(f"Grad on skew params: max={v_test.grad.abs().max():.4e}  "
          f"finite={torch.isfinite(v_test.grad).all().item()}")

    sys.exit(0 if verify_cayley_orthogonality(d=6)["all_pass"] else 1)
