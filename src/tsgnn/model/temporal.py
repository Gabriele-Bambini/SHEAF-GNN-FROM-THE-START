"""
GRU Temporal Evolution of Restriction Maps.

Phase 2B: Implements the GRU that evolves sheaf restriction maps over pseudotime,
modeling how regulatory relationships change along the trajectory.

For each edge e at time t:
    R_e(t) = vec(F_{u,e}(t), F_{v,e}(t))  -- concatenated restriction map params
    c_e(t) = cat(x_u(t), x_v(t))           -- context from endpoint node features
    h_e(t+1) = GRU(cat(R_e(t), c_e(t)), hidden=R_e(t))

The GRU is SHARED across all edges for parameter efficiency.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalRestrictionEvolution(nn.Module):
    """
    GRU-based temporal evolution of sheaf restriction maps.

    A single shared GRU processes all edges, taking as input:
    - Flattened current restriction maps for the edge: (2 * d * d,)
    - Concatenated endpoint node features: (2 * d,)

    And producing updated restriction map parameters: (2 * d * d,)

    Args:
        stalk_dim: d, dimension of sheaf stalks.
        hidden_dim: GRU hidden dimension (defaults to 2 * d^2).
    """

    def __init__(self, stalk_dim: int, hidden_dim: int = 0, n_steps: int = 4):
        super().__init__()
        self.d = stalk_dim
        self.map_dim = 2 * stalk_dim * stalk_dim  # Flattened restriction maps per edge
        self.n_steps = n_steps

        if hidden_dim <= 0:
            hidden_dim = self.map_dim

        self.hidden_dim = hidden_dim

        # Input: concatenated restriction maps + endpoint features
        input_dim = self.map_dim + 2 * stalk_dim

        # Shared GRU cell across all edges
        self.gru = nn.GRUCell(
            input_size=input_dim,
            hidden_size=self.map_dim,
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.map_dim)

    def forward(
        self,
        restriction_maps: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evolve restriction maps one temporal step.

        Args:
            restriction_maps: (E, 2, d, d) current restriction maps.
            node_features: (N, d) current node features in stalk space.
            edge_index: (2, E) edge indices.
            delta_t: (E, 1) or scalar, time elapsed.

        Returns:
            new_restriction_maps: (E, 2, d, d) updated maps.
        """
        E = restriction_maps.shape[0]
        d = self.d

        src = edge_index[0]
        tgt = edge_index[1]

        # Flatten restriction maps per edge: (E, 2*d*d)
        r_flat = restriction_maps.reshape(E, -1)

        # Context: endpoint node features concatenated
        x_src = node_features[src]
        x_tgt = node_features[tgt]
        context = torch.cat([x_src, x_tgt], dim=-1)  # (E, 2*d)

        if delta_t is None:
            dt = torch.ones(E, 1, device=restriction_maps.device, dtype=restriction_maps.dtype)
        else:
            dt = delta_t.expand(E, 1).to(restriction_maps.dtype).to(restriction_maps.device)

        # ── Neural ODE Euler Sub-stepping ──
        # Instead of one step dt, we do n_steps of dt/n_steps
        h = r_flat
        sub_dt = dt / self.n_steps
        
        for _ in range(self.n_steps):
            # GRU input: concatenated current map params + context
            gru_input = torch.cat([h, context], dim=-1)
            
            # Compute unit-time target state
            target_h = self.gru(gru_input, h)
            
            # Euler step: dh/dt = (target_h - h)
            # h = h + sub_dt * (target_h - h)
            h = h + sub_dt * (target_h - h)

        # Layer normalization for training stability
        h = self.layer_norm(h)

        # Reshape back to (E, 2, d, d)
        new_maps = h.reshape(E, 2, d, d)

        return new_maps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test
    E, N, d = 200, 50, 4
    edge_index = torch.randint(0, N, (2, E))
    maps = torch.randn(E, 2, d, d)
    features = torch.randn(N, d)

    module = TemporalRestrictionEvolution(stalk_dim=d)
    new_maps = module(maps, features, edge_index)
    print(f"Input maps: {maps.shape}, Output maps: {new_maps.shape}")
    print(f"Parameter count: {sum(p.numel() for p in module.parameters()):,}")
