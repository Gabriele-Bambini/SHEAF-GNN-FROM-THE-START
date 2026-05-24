"""
B2: Sheaf Neural ODE — continuous-pseudotime restriction map evolution.

Replaces discrete GRU evolution (one step per bin) with a Neural ODE:
    dR/dτ = f_θ(R(τ), x_endpoints)
integrated from τ=0 to τ=δt for each pseudotime bin.

Why Neural ODE > GRU for cross-dataset transfer:
  - GRU learns dataset-specific discrete dynamics (BRCA bins ≠ CRC bins)
  - ODE learns CONTINUOUS dynamics: dR/dτ doesn't depend on bin grid
  - τ normalized to [0, 1] across datasets → BRCA-trained dynamics
    can integrate over CRC's pseudotime without seeing different bin counts
  - Adaptive step size handles variable bin spacing (e.g., DPT non-uniform)

Drop-in replacement for TemporalRestrictionEvolution.forward signature:
    forward(restriction_maps, node_features, edge_index, delta_t) → new_maps

Implementation choices:
  - Endpoint features `context` held constant during integration of one bin
    (x is exogenous control signal, observed only at bins, not interpolated
    inside the integration step — simpler, no autograd issues with
    searchsorted, more stable on CPU).
  - Solver: `rk4` (fixed-step 4-stage Runge-Kutta) — deterministic,
    fast on CPU, exact gradient via reverse-mode autodiff. No need for
    adjoint method since per-bin integration is shallow.
  - Velocity field zero-initialized at output → dR/dτ ≈ 0 at start,
    R(δt) ≈ R(0) → training begins from identity flow (stable).

Theoretical advantage (Chen et al. 2018, Liu+ 2024):
  Neural ODEs are universal approximators for continuous flows.
  For sheaf restriction maps R(τ) ∈ ℝ^{E×2×d×d}, this gives a
  parametric homotopy from initial maps R_0 to evolved R(τ),
  which is exactly the structure of cellular sheaf evolution
  on a stratified pseudotime axis.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["SheafNeuralODE"]


class SheafNeuralODE(nn.Module):
    """
    Neural ODE for restriction map evolution.

    For each edge e=(u,v):
        R_e(τ + δt) = R_e(τ) + ∫_τ^{τ+δt} f_θ(R_e(s), [x_u, x_v]) ds

    f_θ is a 3-layer MLP. Endpoint features [x_u, x_v] are held constant
    during one bin's integration (x evolves bin-by-bin in the outer loop).

    Args:
        stalk_dim: d.
        hidden_dim: Velocity-field MLP hidden width (default 256).
        solver: torchdiffeq solver name ("rk4", "dopri5", "euler").
                Default "rk4" — fixed-step, deterministic, cheap on CPU.
        rtol, atol: Tolerance (used only by adaptive solvers like dopri5).
        n_eval_steps: For fixed-step solvers, number of intermediate
                      evaluation points (default 4 → matches the
                      legacy GRU's n_steps for fair comparison).
    """

    def __init__(
        self,
        stalk_dim: int,
        hidden_dim: int = 256,
        solver: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        n_eval_steps: int = 4,
    ):
        super().__init__()
        self.d = stalk_dim
        self.map_dim = 2 * stalk_dim * stalk_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.n_eval_steps = max(2, int(n_eval_steps))

        # Velocity field: f_θ(R_flat, [x_u, x_v]) → dR/dτ
        # Input:  map_dim + 2*d  (flattened R + endpoint features)
        # Output: map_dim        (dR/dτ)
        input_dim = self.map_dim + 2 * stalk_dim
        self.velocity_field = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.map_dim),
        )

        # Layer norm on the integrated state (matches GRU baseline)
        self.layer_norm = nn.LayerNorm(self.map_dim)

        self._init_parameters()

    def _init_parameters(self):
        """Zero-init last layer → dR/dτ ≈ 0 at start → stable identity flow."""
        last = self.velocity_field[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _ode_rhs(self, t: torch.Tensor, R_flat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        ODE right-hand side: dR/dτ = f_θ(R, context).

        Args:
            t: scalar (unused — autonomous system; pseudotime grid only
               affects integration extent, not the velocity field).
            R_flat: (E, map_dim).
            context: (E, 2*d) endpoint features (held constant per bin).

        Returns:
            dR/dτ: (E, map_dim).
        """
        ode_input = torch.cat([R_flat, context], dim=-1)
        return self.velocity_field(ode_input)

    def forward(
        self,
        restriction_maps: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrate dR/dτ from τ=0 to τ=δt.

        Args:
            restriction_maps: (E, 2, d, d) — R(τ).
            node_features:    (N, d).
            edge_index:       (2, E).
            delta_t:          scalar Tensor or None. Bin width on
                              normalized pseudotime axis. Defaults to 1.0.

        Returns:
            new_maps: (E, 2, d, d) — R(τ + δt).
        """
        # Lazy import — only loaded when ODE actually used
        from torchdiffeq import odeint

        E, _, d, _ = restriction_maps.shape
        device = restriction_maps.device
        dtype = restriction_maps.dtype

        # Flatten R for ODE state
        R_flat = restriction_maps.reshape(E, -1)  # (E, map_dim)

        # Endpoint context (held constant during this bin's integration)
        src, tgt = edge_index[0], edge_index[1]
        context = torch.cat([node_features[src], node_features[tgt]], dim=-1)  # (E, 2*d)

        # Integration extent
        if delta_t is None:
            dt = 1.0
        elif isinstance(delta_t, torch.Tensor):
            # Use mean for global step size (avoid per-edge ODE — too expensive)
            dt = float(delta_t.detach().mean().item()) if delta_t.numel() > 0 else 1.0
        else:
            dt = float(delta_t)
        dt = max(dt, 1e-3)  # avoid zero-length integration (degenerate ODE)

        # Time grid: start, intermediate points (for fixed-step solvers), end
        t_grid = torch.linspace(0.0, dt, self.n_eval_steps, device=device, dtype=dtype)

        # Closure capturing context
        def ode_func(t, R):
            return self._ode_rhs(t, R, context)

        # Integrate
        try:
            if self.solver in ("dopri5", "dopri8", "bosh3"):
                R_traj = odeint(
                    ode_func, R_flat, t_grid,
                    method=self.solver, rtol=self.rtol, atol=self.atol,
                )
            else:
                R_traj = odeint(ode_func, R_flat, t_grid, method=self.solver)
        except Exception as exc:  # pragma: no cover — solver fallback
            logger.warning(f"SheafNeuralODE: solver '{self.solver}' failed ({exc}); falling back to euler")
            R_traj = odeint(ode_func, R_flat, t_grid, method="euler")

        # Take final state
        R_final = R_traj[-1]  # (E, map_dim)

        # LayerNorm removed: it destroys the near-orthogonal structure of R,
        # forcing QR to reconstruct from a near-zero mean matrix instead of
        # perturbing an already-orthogonal one. Kills gradient flow through O(d).

        # Reshape back to (E, 2, d, d)
        new_maps = R_final.reshape(E, 2, d, d)
        return new_maps


# ── Self-contained verification ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    N, E, d = 10, 30, 4
    edge_index = torch.randint(0, N, (2, E))
    R0 = torch.randn(E, 2, d, d)
    x = torch.randn(N, d)

    ode = SheafNeuralODE(d)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.1))
    print(f"R0: {R0.shape}, R1: {R1.shape}, finite={torch.isfinite(R1).all().item()}")

    # Identity at init: zero-init last layer → R(δt) ≈ layer_norm(R0)
    diff = (R1.reshape(E, -1) - ode.layer_norm(R0.reshape(E, -1))).abs().max().item()
    print(f"Identity flow check (≈0 at init): max_diff = {diff:.2e}")

    # Gradient
    R0g = torch.randn(E, 2, d, d, requires_grad=True)
    Rg = ode(R0g, x, edge_index, delta_t=torch.tensor(0.5))
    Rg.sum().backward()
    print(f"Grad on R0: {R0g.grad.abs().max().item():.4e}, finite={torch.isfinite(R0g.grad).all().item()}")

    sys.exit(0)
