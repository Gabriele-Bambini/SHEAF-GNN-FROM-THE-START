"""
FiLM (Feature-wise Linear Modulation) Allele Conditioning.

Phase 2C: Implements FiLM conditioning from ESM-2 allele embeddings.

The allele embedding z_allele tells the model WHICH TP53 mutation is present.
FiLM modulates the GRU output via element-wise scaling and shifting:

    z_allele = MLP(ESM2_embedding)         -- R^1280 -> R^k
    gamma, beta = MLP_film(z_allele)       -- R^k -> R^(2*d*d) each
    h_e(t) = gamma * GRU_output + beta     -- element-wise

Chosen over hypernetwork because:
- FiLM: O(k) parameters vs hypernetwork O(k * p)
- Validated in conditional generation (Perez et al., AAAI 2018)
  and drug-target interaction (HyperPCM, JCIM 2024)
"""

import logging
from typing import Optional  # noqa: F401 (used in forward signatures)

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AlleleFiLM(nn.Module):
    """
    Feature-wise Linear Modulation from ESM-2 allele embeddings.

    Transforms the ESM-2 embedding into gamma (scale) and beta (shift)
    parameters that modulate the restriction map evolution.

    Args:
        esm_dim: Dimension of ESM-2 embeddings (1280 for esm2_t33_650M).
        conditioning_dim: Latent dimension for allele encoding.
        modulation_dim: Dimension of the signal to modulate (2 * d^2 for restriction maps).
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        conditioning_dim: int = 128,
        modulation_dim: int = 32,
    ):
        super().__init__()
        self.modulation_dim = modulation_dim

        # Allele encoder: ESM-2 embedding -> compact representation
        self.allele_encoder = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, conditioning_dim),
            nn.ReLU(),
        )

        # FiLM parameter generator: conditioning -> gamma + beta
        self.film_generator = nn.Sequential(
            nn.Linear(conditioning_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * modulation_dim),  # gamma and beta concatenated
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize so that FiLM starts near identity transform (gamma≈1, beta≈0).

        Small Gaussian weights (std=0.01) on the last film_generator layer ensure:
        - Gradients flow back through FiLM to the ESM embedding (non-zero weights)
        - Different allele embeddings produce measurably different gamma/beta
        - Training starts from a near-identity regime (stable)
        Biases are zeroed so beta starts at 0.
        """
        last_layer = self.film_generator[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        esm_embedding: torch.Tensor,
        gru_output: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,  # ignored; accepted for API compat with NodalAlleleFiLM
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning to GRU output.

        Args:
            esm_embedding: (1, esm_dim) or (esm_dim,) ESM-2 embedding for the allele.
            gru_output: (E, modulation_dim) restriction map updates from GRU.

        Returns:
            Modulated output: (E, modulation_dim).
        """
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)  # (1, esm_dim)

        # Encode allele
        z = self.allele_encoder(esm_embedding)  # (1, conditioning_dim)

        # Generate FiLM parameters
        film_params = self.film_generator(z)  # (1, 2 * modulation_dim)
        gamma, beta = film_params.chunk(2, dim=-1)  # Each: (1, modulation_dim)

        # Initialize gamma around 1 (identity scaling)
        gamma = gamma + 1.0

        # Apply FiLM: element-wise scale and shift
        # gamma and beta are broadcast across all edges (same allele for all)
        modulated = gamma * gru_output + beta  # (E, modulation_dim)

        return modulated

    def get_conditioning_vector(self, esm_embedding: torch.Tensor) -> torch.Tensor:
        """Get the intermediate allele conditioning vector for analysis."""
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)
        return self.allele_encoder(esm_embedding)


class ZeroFiLM(nn.Module):
    """
    Identity FiLM (for ablation: no allele conditioning).

    Always returns gamma=1, beta=0, making the transform an identity.
    """

    def forward(
        self,
        esm_embedding: torch.Tensor,
        gru_output: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,  # ignored; accepted for API compat
    ) -> torch.Tensor:
        return gru_output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test
    E, d = 200, 4
    modulation_dim = 2 * d * d  # 32
    esm_dim = 1280

    film = AlleleFiLM(esm_dim=esm_dim, conditioning_dim=128, modulation_dim=modulation_dim)

    esm_emb = torch.randn(esm_dim)
    gru_out = torch.randn(E, modulation_dim)

    result = film(esm_emb, gru_out)
    print(f"GRU output: {gru_out.shape}, FiLM result: {result.shape}")

    # Verify identity initialization
    film_init = AlleleFiLM(esm_dim=esm_dim, conditioning_dim=128, modulation_dim=modulation_dim)
    result_init = film_init(torch.zeros(esm_dim), gru_out)
    diff = (result_init - gru_out).abs().max()
    print(f"Identity init check (should be ~0): max diff = {diff:.6f}")

    print(f"Parameter count: {sum(p.numel() for p in film.parameters()):,}")
