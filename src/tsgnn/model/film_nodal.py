"""
B1: Nodal FiLM with Laplacian Positional Encoding.

Replaces global FiLM (same γ/β for all edges) with per-node FiLM:
  γ_v, β_v = MLP(z_allele ∥ PE_v)
  h_e = ((γ_src + γ_tgt)/2) * GRU_e + (β_src + β_tgt)/2

Why nodal FiLM is better than global:
  - TP53 mutations only directly perturb downstream targets
    (CDKN1A, MDM2, PUMA, BAX), not upstream regulators
  - Global FiLM bleeds allele signal equally to all 500+ edges
    → noisy predictions, poor cross-dataset transfer
  - Per-node: Laplacian PE anchors node identity in graph topology
    → same positional code for BRCA node i and CRC node i (if same GRN)
    → enables cross-dataset alignment

Laplacian PE:
  Compute normalized graph Laplacian L̂ from edge_index (scalar graph,
  not sheaf). Top-k eigenvectors give stable, graph-topology-based
  positional codes. Cached as buffer (fixed topology).

Integration:
  NodalAlleleFiLM is a drop-in replacement for AlleleFiLM.
  Forward signature: (allele_emb, maps_flat, edge_index) → modulated_maps
  AlleleFiLM.forward updated to accept optional edge_index kwarg (ignored).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["LaplacianPE", "NodalAlleleFiLM"]


class LaplacianPE(nn.Module):
    """
    Positional encoding from normalized graph Laplacian eigenvectors.

    For graph Laplacian L̂ = I - D^{-1/2} A D^{-1/2}, compute top-k
    eigenvectors (Fiedler vectors). These give stable node-level positional
    codes based purely on graph topology — no feature information.

    PE_v = [φ_1(v), φ_2(v), ..., φ_k(v)]   (k-dim vector per node)

    Args:
        num_nodes: N, number of nodes.
        num_eigs: k, number of eigenvectors to use (default 8).
                  For N=500, k=8 captures most structural info.

    Notes:
        - Computed once from edge_index, then cached as buffer.
        - Detached from grad graph (PE is structural, not learned).
        - If N < num_eigs, falls back to identity padding.
    """

    def __init__(self, num_nodes: int, num_eigs: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_eigs = min(num_eigs, num_nodes)
        # Placeholder — filled on first forward call or by precompute()
        self.register_buffer("pe", torch.zeros(num_nodes, self.num_eigs))
        self._computed = False

    def precompute(self, edge_index: torch.Tensor) -> None:
        """Compute and cache Laplacian eigenvectors from edge_index."""
        N = self.num_nodes
        device = edge_index.device

        # Build symmetric adjacency (treat directed as undirected)
        src, tgt = edge_index[0], edge_index[1]
        # Remove self-loops for Laplacian
        mask = src != tgt
        src, tgt = src[mask], tgt[mask]

        # Degree vector
        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, src, torch.ones(src.shape[0], device=device))
        deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=device))
        deg = deg.clamp(min=1.0)

        # Normalized Laplacian: L̂ = I - D^{-1/2} A D^{-1/2}
        # Dense construction (N typically ≤ 500, so fine)
        A = torch.zeros(N, N, device=device)
        A[src, tgt] = 1.0
        A[tgt, src] = 1.0  # symmetric

        D_invsqrt = deg.pow(-0.5)
        L = torch.eye(N, device=device) - D_invsqrt.unsqueeze(1) * A * D_invsqrt.unsqueeze(0)

        # Eigendecomposition (eigh for symmetric matrix → stable, sorted ascending)
        try:
            eigs, vecs = torch.linalg.eigh(L)
            # Skip first eigenvector (eigenvalue ≈ 0, constant vector)
            # Take next num_eigs eigenvectors
            k = min(self.num_eigs, N - 1)
            pe = vecs[:, 1:k + 1]  # (N, k)
            # Pad if needed
            if pe.shape[1] < self.num_eigs:
                pad = torch.zeros(N, self.num_eigs - pe.shape[1], device=device)
                pe = torch.cat([pe, pad], dim=1)
        except RuntimeError:
            logger.warning("LaplacianPE: eigh failed, using zero PE")
            pe = torch.zeros(N, self.num_eigs, device=device)

        self.pe = pe.detach()
        self._computed = True

    def forward(self, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns PE: (N, num_eigs). Computes on first call if not precomputed.

        Args:
            edge_index: (2, E) — required on first call, ignored after caching.
        """
        if not self._computed:
            if edge_index is None:
                raise ValueError("LaplacianPE: edge_index required on first call")
            self.precompute(edge_index)
        return self.pe  # (N, num_eigs)


class NodalAlleleFiLM(nn.Module):
    """
    Per-node FiLM conditioning: allele × graph-position → γ_v, β_v per node.

    Each edge e=(u,v) gets modulation:
        γ_e = (γ_u + γ_v) / 2
        β_e = (β_u + β_v) / 2
        h_e' = γ_e * h_e + β_e

    Drop-in replacement for AlleleFiLM. Extended signature:
        forward(allele_emb, maps_flat, edge_index) → modulated_maps

    Args:
        num_nodes: N.
        esm_dim: ESM-2 embedding dimension (1280).
        conditioning_dim: Latent allele encoding dimension.
        modulation_dim: 2 * d² (flattened restriction maps per edge endpoint pair).
        num_eigs: Laplacian PE dimension (default 8).
    """

    def __init__(
        self,
        num_nodes: int,
        esm_dim: int = 1280,
        conditioning_dim: int = 128,
        modulation_dim: int = 32,
        num_eigs: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.modulation_dim = modulation_dim

        self.laplacian_pe = LaplacianPE(num_nodes, num_eigs=num_eigs)
        pe_dim = self.laplacian_pe.num_eigs

        # Allele encoder: ESM-2 → conditioning_dim
        self.allele_encoder = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, conditioning_dim),
            nn.ReLU(),
        )

        # Node encoder: PE ∥ allele_latent → per-node latent
        self.node_encoder = nn.Sequential(
            nn.Linear(pe_dim + conditioning_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conditioning_dim),
            nn.ReLU(),
        )

        # Per-node FiLM head: conditioning_dim → γ + β (each modulation_dim)
        self.film_head = nn.Linear(conditioning_dim, 2 * modulation_dim)

        self._init_parameters()

    def _init_parameters(self):
        """Init FiLM head near identity (γ≈1, β≈0)."""
        nn.init.normal_(self.film_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.film_head.bias)

    def forward(
        self,
        esm_embedding: torch.Tensor,
        gru_output: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply per-node FiLM to edge-level GRU output.

        Args:
            esm_embedding: (esm_dim,) or (1, esm_dim)
            gru_output: (E, modulation_dim) restriction map updates from GRU
            edge_index: (2, E) — required on first call for PE, cached after

        Returns:
            modulated: (E, modulation_dim)
        """
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)  # (1, esm_dim)

        # 1. Encode allele → (1, conditioning_dim)
        z_allele = self.allele_encoder(esm_embedding)

        # 2. Get Laplacian PE (cached after first call)
        pe = self.laplacian_pe(edge_index)  # (N, num_eigs)

        # 3. Per-node latent: broadcast allele to all nodes
        z_allele_broadcast = z_allele.expand(self.num_nodes, -1)  # (N, conditioning_dim)
        z_node_in = torch.cat([pe, z_allele_broadcast], dim=-1)   # (N, pe_dim+conditioning_dim)
        z_node = self.node_encoder(z_node_in)                       # (N, conditioning_dim)

        # 4. Per-node FiLM params
        film = self.film_head(z_node)                               # (N, 2*modulation_dim)
        gamma_node, beta_node = film.chunk(2, dim=-1)               # Each: (N, modulation_dim)
        gamma_node = gamma_node + 1.0                               # Init near identity

        # 5. Edge-level modulation: average source and target node params
        if edge_index is None:
            raise ValueError("NodalAlleleFiLM: edge_index required")
        src, tgt = edge_index[0], edge_index[1]                     # (E,) each
        gamma_edge = (gamma_node[src] + gamma_node[tgt]) / 2.0     # (E, modulation_dim)
        beta_edge  = (beta_node[src]  + beta_node[tgt])  / 2.0     # (E, modulation_dim)

        # 6. Apply FiLM
        modulated = gamma_edge * gru_output + beta_edge             # (E, modulation_dim)

        return modulated

    def get_conditioning_vector(self, esm_embedding: torch.Tensor) -> torch.Tensor:
        """Allele latent vector (for analysis)."""
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)
        return self.allele_encoder(esm_embedding)


# ── Self-contained verification ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    N, E, d = 20, 60, 4
    esm_dim = 1280
    modulation_dim = 2 * d * d

    edge_index = torch.randint(0, N, (2, E))
    esm_emb = torch.randn(esm_dim)
    gru_out = torch.randn(E, modulation_dim)

    film = NodalAlleleFiLM(N, esm_dim=esm_dim, modulation_dim=modulation_dim)
    out = film(esm_emb, gru_out, edge_index)

    print(f"Input:  {gru_out.shape}")
    print(f"Output: {out.shape}  finite={torch.isfinite(out).all().item()}")

    # Different alleles → different outputs
    esm2 = torch.randn(esm_dim)
    out2 = film(esm2, gru_out, edge_index)
    diff = (out - out2).abs().mean().item()
    print(f"Allele sensitivity (should be >0): {diff:.4f}")

    # Gradient test
    esm_grad = torch.randn(esm_dim, requires_grad=True)
    out_grad = film(esm_grad, gru_out, edge_index)
    out_grad.sum().backward()
    print(f"Gradient on ESM: {esm_grad.grad.abs().max().item():.4e}  finite={torch.isfinite(esm_grad.grad).all().item()}")

    sys.exit(0)
