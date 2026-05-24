"""
Neural Sheaf Diffusion Layer.

Phase 2A: Core sheaf layer based on Bodnar et al. (NeurIPS 2022).

The key insight: cellular sheaves attach vector spaces (stalks) to nodes and edges,
with linear restriction maps encoding how node features relate to shared edge spaces.
The resulting connection Laplacian L_F natively handles heterophily by allowing
sign inversions in restriction maps (modeling transcriptional repression).

Mathematical specification:
- Node v has stalk F(v) = R^d
- Edge e = (u,v) has restriction maps F_{u,e}, F_{v,e}: R^d -> R^d (d x d matrices)
- Sheaf Laplacian L_F (block structure, Nd x Nd):
    Off-diagonal block (u,v): -F_{u,e}^T @ F_{v,e}
    Diagonal block (u,u): sum_{e incident to u} F_{u,e}^T @ F_{u,e}
- Sheaf diffusion: dX/dt = -sigma(L_F @ X @ W)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SheafDiffusionLayer(nn.Module):
    """
    Neural Sheaf Diffusion layer for GRN heterophily resolution.

    For each directed edge e = (u, v), learns two restriction maps:
        F_{u,e}: R^d -> R^d  (source restriction)
        F_{v,e}: R^d -> R^d  (target restriction)
    represented as d x d matrices.

    These maps allow the connection Laplacian to encode:
    - Positive correlation (activation): restriction maps with positive eigenvalues
    - Negative correlation (repression): restriction maps with negative eigenvalues / sign inversions

    Args:
        num_nodes: N, number of nodes in the graph.
        num_edges: E, number of directed edges.
        stalk_dim: d, dimension of each node's stalk space.
        edge_index: (2, E) tensor of directed edge indices.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
        laplacian_eps: float = 1e-3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.d = stalk_dim
        self.register_buffer("edge_index", edge_index)

        # Learnable restriction maps: (E, 2, d, d)
        # restriction_maps[:, 0, :, :] = F_{u,e} (source)
        # restriction_maps[:, 1, :, :] = F_{v,e} (target)
        self.restriction_maps = nn.Parameter(
            torch.empty(num_edges, 2, stalk_dim, stalk_dim)
        )

        # Learnable weight matrix for diffusion: (d, d)
        self.weight = nn.Parameter(torch.empty(stalk_dim, stalk_dim))

        # Activation
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # Diagonal PSD shift to ensure non-negative eigenvalues. Set to 0 in
        # tests verifying exact reduction to graph Laplacian at d=1.
        self._laplacian_eps: float = float(laplacian_eps)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize restriction maps near orthogonal and weight matrix."""
        # Orthogonal initialization for restriction maps
        # This ensures the sheaf starts as a well-conditioned structure
        for e in range(self.num_edges):
            for s in range(2):
                nn.init.orthogonal_(self.restriction_maps.data[e, s])

        # Xavier initialization for weight matrix
        nn.init.xavier_uniform_(self.weight)

    def compute_connection_laplacian(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the (N*d, N*d) sheaf connection Laplacian.

        For nodes u, v connected by edge e = (u, v):
            L_F[u*d:(u+1)*d, v*d:(v+1)*d] = -F_{u,e}^T @ F_{v,e}
            L_F[u*d:(u+1)*d, u*d:(u+1)*d] += F_{u,e}^T @ F_{u,e}  (diagonal)

        For N=500, d=4, this is a 2000x2000 dense matrix (manageable).

        Args:
            restriction_maps: Optional external restriction maps (E, 2, d, d).
                              If None, uses self.restriction_maps.

        Returns:
            L_F: (N*d, N*d) sheaf Laplacian tensor.
        """
        if restriction_maps is None:
            maps = self.restriction_maps
        else:
            maps = restriction_maps

        N, d = self.num_nodes, self.d
        Nd = N * d
        device = maps.device

        # Initialize Laplacian as zeros
        L = torch.zeros(Nd, Nd, device=device)

        src = self.edge_index[0]  # (E,)
        tgt = self.edge_index[1]  # (E,)

        F_src = maps[:, 0]  # (E, d, d) - source restriction maps
        F_tgt = maps[:, 1]  # (E, d, d) - target restriction maps

        # Off-diagonal blocks: L[u_block, v_block] = -F_{u,e}^T @ F_{v,e}
        # Diagonal blocks: L[u_block, u_block] += F_{u,e}^T @ F_{u,e}
        #                   L[v_block, v_block] += F_{v,e}^T @ F_{v,e}

        # Compute all block products at once
        # F_src^T @ F_tgt: (E, d, d)
        off_diag = -torch.bmm(F_src.transpose(1, 2), F_tgt)  # (E, d, d)

        # Diagonal contributions from source side: F_src^T @ F_src
        diag_src = torch.bmm(F_src.transpose(1, 2), F_src)  # (E, d, d)

        # Diagonal contributions from target side: F_tgt^T @ F_tgt
        diag_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)  # (E, d, d)

        # Scatter into the Laplacian matrix
        for e_idx in range(self.num_edges):
            u = src[e_idx].item()
            v = tgt[e_idx].item()

            u_start, u_end = u * d, (u + 1) * d
            v_start, v_end = v * d, (v + 1) * d

            # Off-diagonal: (u, v) block
            L[u_start:u_end, v_start:v_end] += off_diag[e_idx]
            # Off-diagonal: (v, u) block (symmetric)
            L[v_start:v_end, u_start:u_end] += off_diag[e_idx].T

            # Diagonal contributions
            L[u_start:u_end, u_start:u_end] += diag_src[e_idx]
            L[v_start:v_end, v_start:v_end] += diag_tgt[e_idx]

        # Diagonal PSD shift (ε·I). Arbitrary restriction maps can yield
        # numerically negative eigenvalues → diffusion amplification → NaN.
        if self._laplacian_eps > 0:
            L = L + self._laplacian_eps * torch.eye(Nd, device=device, dtype=L.dtype)

        return L


    def diffuse(
        self,
        x: torch.Tensor,
        L_F: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one step of sheaf diffusion.

        dX/dt = -sigma(L_F @ X_flat @ W)

        where X_flat is x reshaped to (N*d,) for block multiplication.

        Args:
            x: Node features (N, d).
            L_F: Sheaf Laplacian (N*d, N*d).

        Returns:
            Updated node features (N, d).
        """
        N, d = x.shape
        assert d == self.d, f"Expected stalk_dim {self.d}, got {d}"

        # Reshape to block vector: (N*d,)
        x_flat = x.reshape(-1)  # (N*d,)

        # Apply Laplacian: L_F @ x_flat -> (N*d,)
        Lx = L_F @ x_flat  # (N*d,)

        # Reshape back to (N, d) and apply weight + activation
        Lx = Lx.reshape(N, d)
        out = -self.activation(Lx @ self.weight)

        # Residual connection
        return x + out

    def forward(
        self,
        x: torch.Tensor,
        restriction_maps: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: compute Laplacian and run sheaf diffusion.

        Args:
            x: Node features (N, d).
            restriction_maps: Optional external maps (E, 2, d, d).
            num_steps: Number of diffusion steps.

        Returns:
            x_out: Updated node features (N, d).
            L_F: Sheaf Laplacian (N*d, N*d).
        """
        L_F = self.compute_connection_laplacian(restriction_maps)

        for _ in range(num_steps):
            x = self.diffuse(x, L_F)

        return x, L_F


def verify_sheaf_reduces_to_graph_laplacian(num_nodes=5, stalk_dim=2):
    """
    Sanity check: with identity restriction maps, the sheaf Laplacian
    should reduce to the standard graph Laplacian (block-diagonal form).
    """
    import networkx as nx

    # Create a simple graph
    G = nx.cycle_graph(num_nodes)
    edges = list(G.edges())
    E = len(edges)
    edge_index = torch.tensor([[u for u, v in edges], [v for u, v in edges]], dtype=torch.long)

    layer = SheafDiffusionLayer(num_nodes, E, stalk_dim, edge_index)

    # Set restriction maps to identity
    with torch.no_grad():
        layer.restriction_maps.fill_(0)
        for e in range(E):
            for s in range(2):
                for i in range(stalk_dim):
                    layer.restriction_maps[e, s, i, i] = 1.0

    L_F = layer.compute_connection_laplacian()

    # The standard graph Laplacian for a cycle: degree - adjacency
    L_graph = nx.laplacian_matrix(G).toarray()

    # L_F should be block-diagonal with L_graph replicated d times
    # Check: L_F[i*d, j*d] should equal L_graph[i, j] for all i, j
    L_check = L_F[::stalk_dim, ::stalk_dim].detach().numpy()

    error = abs(L_check - L_graph).max()
    logger.info(f"Sheaf-to-graph Laplacian verification: max error = {error:.6f}")
    assert error < 1e-5, f"Verification failed: max error {error}"
    logger.info("PASSED: Sheaf Laplacian reduces to graph Laplacian with identity maps")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Run sanity check
    verify_sheaf_reduces_to_graph_laplacian(num_nodes=5, stalk_dim=2)
    verify_sheaf_reduces_to_graph_laplacian(num_nodes=10, stalk_dim=4)

    # Test with random graph
    N, E, d = 50, 200, 4
    edge_index = torch.randint(0, N, (2, E))
    layer = SheafDiffusionLayer(N, E, d, edge_index)
    x = torch.randn(N, d)
    x_out, L_F = layer(x, num_steps=3)
    print(f"Input: {x.shape}, Output: {x_out.shape}, Laplacian: {L_F.shape}")
    print(f"Laplacian symmetric: {torch.allclose(L_F, L_F.T, atol=1e-5)}")
