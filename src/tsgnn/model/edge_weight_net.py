"""
TSGNN-v2 Step 2: Edge Weight Network — separazione trasporto / intensità.

Problem fixed
-------------
In v1 the allele signal (ESM) modulated the restriction maps *before* QR,
making it non-identifiable.  In v2:

    Q_e  ∈ O(d)  — geometric transport   (from Cayley params, ODE output)
    w_e  > 0     — regulatory strength   (from this module, ESM-conditioned)

The weighted connection Laplacian is then:

    (δ_Q x)_e = Q_{u,e} x_u − Q_{v,e} x_v
    L_t = δ_Q^T · diag(w_e ⊗ I_d) · δ_Q

This separates:
  - *Transport*: which direction a signal propagates (geometric, ODE-learned)
  - *Strength*:  how strongly an edge contributes (biological, ESM-modulated)

ESM conditioning is NOW applied to w_e — a scalar that is NEVER projected
away — so the allele signal reaches the Laplacian with 100% identifiability.

Architecture
------------
    context_e = [x_u(t), x_v(t), z_esm]          (2d + esm_latent dim)
    logw_e    = MLP_w(context_e)                   (scalar per edge)
    w_e       = softplus(logw_e) + ε               (positive, bounded below)

Optional: full d×d matrix weight W_e = B_e^T B_e + ε·I (richer but slower).

Usage
-----
    net = EdgeWeightNet(stalk_dim=6, esm_latent_dim=128)
    w   = net(x, esm_latent, edge_index)   # (E,) positive weights
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "EdgeWeightNet",
    "WeightedConnectionLaplacian",
    "WeightedNormalizedOrthogonalLaplacian",
]


class EdgeWeightNet(nn.Module):
    """
    MLP that produces positive edge weights conditioned on node features
    and the allele ESM latent vector.

    Args:
        stalk_dim:       d — node feature dimension in stalk space.
        esm_latent_dim:  Dimension of the compressed ESM allele representation.
                         (The raw 1280-d ESM vector should be compressed
                          externally, e.g. by AlleleFiLM.allele_encoder.)
        hidden_dim:      MLP hidden width.
        eps:             Minimum weight value (numerical stability).
        use_matrix_weight: If True, produce d×d PSD matrices instead of scalars.
                           Richer but d² more expensive. Default False.
    """

    def __init__(
        self,
        stalk_dim: int,
        esm_latent_dim: int = 128,
        hidden_dim: int = 128,
        eps: float = 1e-3,
        use_matrix_weight: bool = False,
    ):
        super().__init__()
        self.d   = stalk_dim
        self.eps = eps
        self.use_matrix_weight = use_matrix_weight

        # Input: [x_u, x_v, z_esm]
        in_dim = 2 * stalk_dim + esm_latent_dim

        # Shared edge encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        if use_matrix_weight:
            # B_e ∈ R^{d×d}  →  W_e = B_e^T B_e + ε I  (PSD)
            self.head = nn.Linear(hidden_dim, stalk_dim * stalk_dim)
        else:
            # Scalar weight per edge
            self.head = nn.Linear(hidden_dim, 1)

        self._init_parameters()

    def _init_parameters(self):
        """Zero-init output head → all weights start near ε (unbiased)."""
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        esm_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute positive edge weights.

        Args:
            x:          (N, d) node features in stalk space.
            esm_latent: (conditioning_dim,) or (1, conditioning_dim) allele latent.
            edge_index: (2, E) directed edge indices.

        Returns:
            If use_matrix_weight=False:  (E,) scalar weights, all > 0.
            If use_matrix_weight=True:   (E, d, d) PSD matrices.
        """
        src, tgt = edge_index[0], edge_index[1]   # (E,)

        # Node endpoint features
        x_u = x[src]   # (E, d)
        x_v = x[tgt]   # (E, d)

        # Broadcast ESM latent to all edges
        if esm_latent.dim() == 1:
            esm_latent = esm_latent.unsqueeze(0)   # (1, esm_latent_dim)
        E = x_u.shape[0]
        z_esm = esm_latent.expand(E, -1)           # (E, esm_latent_dim)

        # Edge context
        context = torch.cat([x_u, x_v, z_esm], dim=-1)   # (E, 2d+esm_latent_dim)

        h = self.encoder(context)                          # (E, hidden_dim)
        out = self.head(h)                                 # (E, 1) or (E, d*d)

        if self.use_matrix_weight:
            B = out.reshape(E, self.d, self.d)             # (E, d, d)
            W = torch.bmm(B.transpose(1, 2), B)            # (E, d, d)  B^T B
            I = torch.eye(self.d, device=W.device, dtype=W.dtype).unsqueeze(0)
            return W + self.eps * I                        # (E, d, d) PSD

        # Scalar: softplus ensures positivity
        w = F.softplus(out.squeeze(-1)) + self.eps        # (E,)
        return w


class WeightedConnectionLaplacian(nn.Module):
    """
    Builds the weighted sheaf Laplacian:

        L = δ_Q^T · W · δ_Q

    where:
        δ_Q is the coboundary operator with orthogonal maps Q_e.
        W = diag(w_e ⊗ I_d)  for scalar weights  w_e > 0.

    This generalizes the unweighted Laplacian (w_e=1 for all e) and
    ensures L ≽ 0 by construction (regardless of the Q_e values).

    Args:
        num_nodes:  N.
        stalk_dim:  d.
        edge_index: (2, E) buffer (registered, not learned).
    """

    def __init__(
        self,
        num_nodes: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
        laplacian_eps: float = 1e-3,
    ):
        super().__init__()
        self.N   = num_nodes
        self.d   = stalk_dim
        self.eps = laplacian_eps
        self.register_buffer("edge_index", edge_index)

    def forward(
        self,
        orthogonal_maps: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the (Nd, Nd) weighted sheaf Laplacian.

        Args:
            orthogonal_maps: (E, 2, d, d) — Q_{u,e} and Q_{v,e} for each edge.
            edge_weights:    (E,) scalar weights (default: all ones).

        Returns:
            L: (Nd, Nd) PSD matrix.
        """
        N, d = self.N, self.d
        Nd   = N * d
        E    = orthogonal_maps.shape[0]
        dev  = orthogonal_maps.device
        dtype = orthogonal_maps.dtype

        src = self.edge_index[0]   # (E,)
        tgt = self.edge_index[1]   # (E,)

        Q_src = orthogonal_maps[:, 0]   # (E, d, d)
        Q_tgt = orthogonal_maps[:, 1]   # (E, d, d)

        if edge_weights is None:
            w = torch.ones(E, device=dev, dtype=dtype)
        else:
            w = edge_weights.to(device=dev, dtype=dtype)   # (E,)

        # Weighted blocks:
        #   off-diagonal (u,v): -w_e * Q_{u,e}^T Q_{v,e}
        #   diagonal u:         +w_e * Q_{u,e}^T Q_{u,e} = w_e * I  (since Q ortho)
        #   diagonal v:         +w_e * I

        w_bc = w.view(E, 1, 1)   # (E, 1, 1) for broadcasting

        off_diag  = -w_bc * torch.bmm(Q_src.transpose(1, 2), Q_tgt)   # (E, d, d)
        diag_src  =  w_bc * torch.eye(d, device=dev, dtype=dtype).unsqueeze(0).expand(E, d, d)
        diag_tgt  =  w_bc * torch.eye(d, device=dev, dtype=dtype).unsqueeze(0).expand(E, d, d)

        L = torch.zeros(Nd, Nd, device=dev, dtype=dtype)

        # Vectorized scatter — build block indices
        src_np = src.tolist()
        tgt_np = tgt.tolist()

        for e in range(E):
            u, v = src_np[e], tgt_np[e]
            us, ue = u * d, (u + 1) * d
            vs, ve = v * d, (v + 1) * d

            L[us:ue, vs:ve] += off_diag[e]
            L[vs:ve, us:ue] += off_diag[e].T

            L[us:ue, us:ue] += diag_src[e]
            L[vs:ve, vs:ve] += diag_tgt[e]

        # PSD regularization
        if self.eps > 0:
            L = L + self.eps * torch.eye(Nd, device=dev, dtype=dtype)

        return L


class WeightedNormalizedOrthogonalLaplacian(nn.Module):
    """
    Build a normalized weighted connection Laplacian for orthogonal maps.

    This is the hardened v3 path. It delegates the block algebra to the
    contract-checked sheaf_ts core builder, which verifies orthogonality,
    computes D^{-1/2} L D^{-1/2}, and does not add eps * I to the
    Laplacian kernel.
    """

    def __init__(
        self,
        num_nodes: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
        degree_regularizer: float = 1e-6,
        orthogonal_atol: float = 1e-5,
    ):
        super().__init__()
        self.N = num_nodes
        self.d = stalk_dim
        self.degree_regularizer = degree_regularizer
        self.orthogonal_atol = orthogonal_atol
        self.register_buffer("edge_index", edge_index)

    def forward(
        self,
        orthogonal_maps: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_weights is not None and edge_weights.dim() != 1:
            raise ValueError(
                "WeightedNormalizedOrthogonalLaplacian currently accepts scalar "
                "edge_weights with shape (E,)"
            )
        from sheaf_ts.core.orthogonal import orthogonality_error

        err = orthogonality_error(orthogonal_maps)
        if err > self.orthogonal_atol:
            raise ValueError(
                "connection Laplacian requires orthogonal maps; "
                f"max Q.T@Q-I error is {err:.3e}"
            )

        num_edges = orthogonal_maps.shape[0]
        dev = orthogonal_maps.device
        dtype = orthogonal_maps.dtype
        edge_index = self.edge_index.to(device=dev, dtype=torch.long)
        src = edge_index[0]
        tgt = edge_index[1]
        if edge_weights is None:
            weights = torch.ones(num_edges, device=dev, dtype=dtype)
        else:
            weights = edge_weights.to(device=dev, dtype=dtype)

        degree = torch.zeros(self.N, device=dev, dtype=dtype)
        degree.index_add_(0, src, weights)
        degree.index_add_(0, tgt, weights)
        norm_degree = (degree + self.degree_regularizer).clamp_min(1e-12)
        inv_sqrt = norm_degree.rsqrt()

        q_src = orthogonal_maps[:, 0]
        q_tgt = orthogonal_maps[:, 1]
        eye = torch.eye(self.d, device=dev, dtype=dtype).expand(num_edges, self.d, self.d)
        diag_src = (weights / norm_degree[src]).view(num_edges, 1, 1) * eye
        diag_tgt = (weights / norm_degree[tgt]).view(num_edges, 1, 1) * eye
        off_scale = (-weights * inv_sqrt[src] * inv_sqrt[tgt]).view(num_edges, 1, 1)
        off_src_tgt = off_scale * torch.bmm(q_src.transpose(1, 2), q_tgt)
        off_tgt_src = off_src_tgt.transpose(1, 2)

        blocks = torch.zeros(
            self.N * self.N,
            self.d,
            self.d,
            device=dev,
            dtype=dtype,
        )
        blocks.index_add_(0, src * self.N + src, diag_src)
        blocks.index_add_(0, tgt * self.N + tgt, diag_tgt)
        blocks.index_add_(0, src * self.N + tgt, off_src_tgt)
        blocks.index_add_(0, tgt * self.N + src, off_tgt_src)
        laplacian = blocks.view(self.N, self.N, self.d, self.d).permute(
            0, 2, 1, 3
        ).reshape(self.N * self.d, self.N * self.d)
        return 0.5 * (laplacian + laplacian.T)


# ── Verification ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    N, E, d = 15, 50, 6
    esm_latent_dim = 128
    edge_index = torch.randint(0, N, (2, E))

    # Node features and allele latent
    x         = torch.randn(N, d)
    esm_latent = torch.randn(esm_latent_dim)

    # Edge weight net
    net = EdgeWeightNet(stalk_dim=d, esm_latent_dim=esm_latent_dim)
    w   = net(x, esm_latent, edge_index)
    print(f"Edge weights: {w.shape}  min={w.min():.4f}  max={w.max():.4f}  all_pos={bool((w > 0).all())}")

    # Weighted Laplacian
    from cayley_orthogonal import skew_to_orthogonal  # noqa: local test import
    k    = d * (d - 1) // 2
    v    = torch.randn(E, 2, k)
    flat = v.reshape(E * 2, k)
    Q    = skew_to_orthogonal(flat, d).reshape(E, 2, d, d)

    laplacian_builder = WeightedConnectionLaplacian(N, d, edge_index)
    L = laplacian_builder(Q, w)

    # Verify PSD
    eigs = torch.linalg.eigvalsh(L.float())
    print(f"L: {L.shape}  min_eig={eigs.min():.4f}  symmetric={torch.allclose(L, L.T, atol=1e-5)}")
    print(f"L PSD: {bool(eigs.min() > -1e-4)}")

    # Gradient check: ESM → weights → Laplacian
    esm_grad = torch.randn(esm_latent_dim, requires_grad=True)
    w2  = net(x, esm_grad, edge_index)
    L2  = laplacian_builder(Q, w2)
    L2.sum().backward()
    print(f"Grad on ESM latent: max={esm_grad.grad.abs().max():.4e}  "
          f"finite={torch.isfinite(esm_grad.grad).all().item()}")

    sys.exit(0)
