"""
Full TS-GNN (Temporal Sheaf Graph Neural Network) Architecture.

Phase 2D: Assembles sheaf diffusion + GRU temporal evolution + FiLM conditioning
into the complete model that maps (temporal_graphs, allele_embedding) -> predictions.

Architecture flow at each time step t:
1. Project input features to stalk space: x = proj(X(t))
2. Evolve restriction maps via GRU: R(t+1) = GRU(R(t), context)
3. Apply FiLM conditioning from allele: R'(t+1) = gamma * R(t+1) + beta
4. Compute connection Laplacian: L_F = sheaf_laplacian(R'(t+1))
5. Sheaf diffusion (multiple steps): x = diffuse(x, L_F)
6. Project back to expression space: X_hat(t+1) = proj_out(x)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .sheaf import SheafDiffusionLayer
from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion
from .sheaf_sparse import SparseVectorizedSheafDiffusion
from .diffusion_implicit import ImplicitOrthogonalSheafDiffusion
from .cayley_orthogonal import CayleyOrthogonalSheafDiffusion  # v2
from .edge_weight_net import (
    EdgeWeightNet,
    WeightedConnectionLaplacian,
    WeightedNormalizedOrthogonalLaplacian,
)  # v2/v3
from .temporal import TemporalRestrictionEvolution
from .temporal_ode import SheafNeuralODE

logger = logging.getLogger(__name__)


class StalkInterpreter(nn.Module):
    """
    Grounds latent stalk dimensions into biological entities.
    Default mapping for d=4:
    0: TF Activity (Regulator potential)
    1: Epigenetic Accessibility (Gatekeeper)
    2: mRNA Stability (Decay rate)
    3: Protein Abundance (Translation proxy)
    """
    def __init__(self, stalk_dim: int):
        super().__init__()
        self.d = stalk_dim
        # Learnable mapping to ground dimensions, initialized as identity
        self.projection = nn.Linear(stalk_dim, stalk_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(stalk_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def get_dimension_name(self, dim: int) -> str:
        names = ["TF Activity", "Accessibility", "Stability", "Abundance"]
        return names[dim] if dim < len(names) else f"Latent_{dim}"


class ZeroAlleleEncoder(nn.Module):
    """Allele-invariant conditioning vector for no-allele ablations."""

    def __init__(self, conditioning_dim: int):
        super().__init__()
        self.conditioning_dim = conditioning_dim

    def forward(self, allele_embedding: torch.Tensor) -> torch.Tensor:
        if allele_embedding.dim() == 1:
            return allele_embedding.new_zeros(self.conditioning_dim)
        return allele_embedding.new_zeros(allele_embedding.shape[0], self.conditioning_dim)


class TSGNN(nn.Module):
    """
    Temporal Sheaf Graph Neural Network for allele-conditioned GRN rewiring.

    Args:
        num_nodes: N, number of genes/TFs.
        num_edges: E, number of regulatory edges.
        stalk_dim: d, sheaf stalk dimension.
        input_dim: Dimension of input gene features.
        esm_dim: Dimension of ESM-2 allele embeddings.
        conditioning_dim: FiLM conditioning latent dimension.
        edge_index: (2, E) directed edge tensor.
        num_diffusion_steps: Sheaf diffusion iterations per time step.
        use_allele_conditioning: If False, disables FiLM (ablation).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        input_dim: int,
        esm_dim: int,
        conditioning_dim: int,
        edge_index: torch.Tensor,
        num_diffusion_steps: int = 3,
        use_allele_conditioning: bool = True,
        use_vectorized: bool = True,
        use_gradient_checkpointing: bool = False,
        use_normalized_laplacian: bool = True,
        use_implicit_diffusion: bool = True,
        predict_delta: bool = True,    # v2: predict delta to recover true states
        use_neural_ode: bool = True,
        ode_hidden_dim: int = 256,
        ode_solver: str = "rk4",
        edge_weight_hidden: int = 128, # v2: EdgeWeightNet hidden dim
        **legacy_options,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.d = stalk_dim
        self.input_dim = input_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.use_allele_conditioning = use_allele_conditioning
        self.use_vectorized = use_vectorized
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_normalized_laplacian = use_normalized_laplacian
        self.use_implicit_diffusion = use_implicit_diffusion
        self.predict_delta = predict_delta
        self.use_neural_ode = use_neural_ode
        self.k = stalk_dim * (stalk_dim - 1) // 2
        self.use_skew_temporal_state = use_neural_ode

        self.register_buffer("edge_index", edge_index)

        # Input projection: gene expression -> stalk space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, stalk_dim),
        )

        # ── v2: Cayley parametrization + weighted Laplacian ──────────
        self.sheaf_layer = CayleyOrthogonalSheafDiffusion(
            num_nodes=num_nodes,
            num_edges=num_edges,
            stalk_dim=stalk_dim,
            edge_index=edge_index,
            use_implicit=use_implicit_diffusion,
        )
        # EdgeWeightNet: ESM conditioning goes HERE
        self.edge_weight_net = EdgeWeightNet(
            stalk_dim=stalk_dim,
            esm_latent_dim=conditioning_dim,
            hidden_dim=edge_weight_hidden,
        )
        if use_normalized_laplacian:
            self.weighted_laplacian = WeightedNormalizedOrthogonalLaplacian(
                num_nodes=num_nodes,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        else:
            self.weighted_laplacian = WeightedConnectionLaplacian(
                num_nodes=num_nodes,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
                laplacian_eps=0.0,
            )

        # Temporal evolution of restriction maps
        # B2: use_neural_ode=True → SheafNeuralODE (continuous pseudotime, dataset-agnostic)
        #     use_neural_ode=False → TemporalRestrictionEvolution (legacy GRU)
        if use_neural_ode:
            self.temporal_evolution = SheafNeuralODE(
                stalk_dim=stalk_dim,
                hidden_dim=ode_hidden_dim,
                solver=ode_solver,
                state_dim=2 * self.k,
            )
        else:
            self.temporal_evolution = TemporalRestrictionEvolution(
                stalk_dim=stalk_dim,
            )

        # v2: ESM → conditioning_dim latent (used by EdgeWeightNet)
        if not use_allele_conditioning:
            self.allele_encoder = ZeroAlleleEncoder(conditioning_dim)
        else:
            self.allele_encoder = nn.Sequential(
                nn.Linear(esm_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, conditioning_dim),
                nn.ReLU(),
            )

        # Stalk Interpreter (Interpretability Layer)
        self.interpreter = StalkInterpreter(stalk_dim)

        # Output projection: stalk space -> expression prediction
        self.output_proj = nn.Sequential(
            nn.Linear(stalk_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, input_dim),
        )

        # Initial restriction maps (learned starting point). Orthogonal
        # initialisation ensures the sheaf Laplacian starts well-conditioned:
        # F^T F ≈ I on every edge, so the block-diagonal degree blocks are
        # near-identity and the Cheeger-bounded normalised Laplacian has
        # eigenvalues tightly packed in [0, 2] at step 0.
        self.initial_maps = nn.Parameter(
            torch.empty(num_edges, 2, stalk_dim, stalk_dim)
        )
        # Flatten edge / endpoint axes to (2E, d, d) once, orthogonalise in
        # place, then let PyTorch view it back as (E, 2, d, d) at access time.
        # No reshape round-trip, no redundant copy.
        flat_view = self.initial_maps.data.view(num_edges * 2, stalk_dim, stalk_dim)
        nn.init.orthogonal_(flat_view)
        self.initial_skew = nn.Parameter(torch.zeros(num_edges, 2, self.k))

    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
        edge_weights_seq: Optional[torch.Tensor] = None,
        bin_boundaries: Optional[torch.Tensor] = None,
        return_stalks: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass over the temporal graph sequence.

        Args:
            node_features_seq: (K, N, input_dim) temporal node features.
            allele_embedding: (esm_dim,) or (1, esm_dim) ESM-2 embedding.
            edge_weights_seq: Optional (K, E) temporal edge weights.

        Returns:
            predictions: List of K tensors, each (N, input_dim).
            restriction_maps_trajectory: List of K tensors, each (E, 2, d, d).
            sheaf_laplacians: List of K tensors, each (N*d, N*d).
            stalks_trajectory: Optional list of K tensors, each (N, d), when
                ``return_stalks=True``.
        """
        K = node_features_seq.shape[0]
        predictions = []
        maps_trajectory = []
        laplacians = []
        stalks_trajectory = []

        # Start with packed skew params in the v3 ODE path; legacy GRU keeps raw maps.
        current_maps = (
            self.initial_skew.clone()
            if self.use_skew_temporal_state
            else self.initial_maps.clone()
        )

        def _edge_weight_prior(t_inner: int, like: torch.Tensor) -> Optional[torch.Tensor]:
            if edge_weights_seq is None:
                return None
            prior = edge_weights_seq
            if prior.dim() == 2:
                if prior.shape[0] != K:
                    raise ValueError(
                        f"edge_weights_seq first dimension must match K={K}; got {prior.shape[0]}"
                    )
                prior = prior[t_inner]
            if prior.shape != (self.num_edges,):
                raise ValueError(
                    f"edge_weights_seq step shape must be ({self.num_edges},), got {tuple(prior.shape)}"
                )
            prior = prior.to(device=like.device, dtype=like.dtype).clamp_min(0.0)
            return prior / prior.mean().clamp_min(1e-6)

        def _step(current_maps_inner, node_feat_t, allele_emb_inner, step_idx_inner, delta_t_inner=None):
            # 1. Project input features to stalk space
            x_latent = self.input_proj(node_feat_t)  # (N, d)

            # 2. Evolve restriction maps via ODE/GRU
            new_maps = self.temporal_evolution(
                current_maps_inner, x_latent, self.edge_index, delta_t_inner
            )

            # 3a. Maps from ODE → Cayley-orthogonalized (no QR nullspace)
            orth_maps = self.sheaf_layer.get_orthogonal_maps(
                new_maps,
                allow_qr_fallback=not self.use_skew_temporal_state,
            )

            # 3b. ESM → latent → edge weights (ESM enters HERE, post-ortho)
            z_esm = self.allele_encoder(allele_emb_inner)  # (conditioning_dim,)
            w_e   = self.edge_weight_net(x_latent, z_esm, self.edge_index)  # (E,)
            prior = _edge_weight_prior(int(step_idx_inner.detach().item()), w_e)
            if prior is not None:
                w_e = w_e * prior

            # 4. Weighted connection Laplacian L = δ_Q^T W δ_Q
            L_F   = self.weighted_laplacian(orth_maps, w_e)
            # 5. Sheaf diffusion (Crank-Nicolson via CG)
            x = x_latent
            for _ in range(self.num_diffusion_steps):
                x = self.sheaf_layer.diffuse(x, L_F)

            # 5.5 Grounded interpretation
            x_interpreted = self.interpreter(x)

            # 6. Output projection + residual
            # output_proj predicts ΔX (the change from X(t) to X(t+1)).
            # Final: X̂(t+1) = X(t) + ΔX  when predict_delta=True.
            # MSE(X(t)+ΔX_pred, X(t+1)) = MSE(ΔX_pred, ΔX_true) — same gradient,
            # but model can now learn "no change" by predicting ΔX≈0 (identity baseline).
            delta = self.output_proj(x_interpreted)  # (N, input_dim)
            if self.predict_delta:
                x_pred = node_feat_t + delta
            else:
                x_pred = delta  # legacy: predict absolute X (old behaviour)
            return x_pred, new_maps, orth_maps, L_F, x_interpreted

        for t in range(K):
            delta_t = None
            if bin_boundaries is not None and t < len(bin_boundaries) - 1:
                # Actual duration of time step t
                dt_val = bin_boundaries[t + 1] - bin_boundaries[t]
                # Single scalar value, we let temporal gru handle expansion
                delta_t = torch.tensor(dt_val, dtype=torch.float32, device=node_features_seq.device)

            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory on long sequences
                step_idx = torch.tensor(t, dtype=torch.long, device=node_features_seq.device)
                x_pred, current_maps, orth_maps_t, L_F, stalks_t = grad_ckpt(
                    _step, current_maps, node_features_seq[t], allele_embedding, step_idx, delta_t,
                    use_reentrant=False
                )
            else:
                step_idx = torch.tensor(t, dtype=torch.long, device=node_features_seq.device)
                x_pred, current_maps, orth_maps_t, L_F, stalks_t = _step(
                    current_maps, node_features_seq[t], allele_embedding, step_idx, delta_t
                )

            maps_trajectory.append(orth_maps_t.detach().clone())
            laplacians.append(L_F)
            predictions.append(x_pred)
            stalks_trajectory.append(stalks_t)

        if return_stalks:
            return predictions, maps_trajectory, laplacians, stalks_trajectory
        return predictions, maps_trajectory, laplacians

    def decompose_rewiring(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
        bin_boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Decompose rewiring into context + allele-specific components.

        L_F(t) = L_F^context(t) + Delta_L_F^allele(t)

        Runs forward twice:
        1. With allele embedding (full model)
        2. With zero allele embedding (context only)

        Returns:
            context_laplacians: List of K Laplacians from context-only model.
            allele_laplacians: List of K full Laplacians.
            delta_laplacians: List of K (allele - context) perturbation Laplacians.
        """
        # Full model
        out_full = self.forward(
            node_features_seq, allele_embedding, bin_boundaries=bin_boundaries
        )
        _, _, full_laplacians = out_full

        # Context only (zero allele embedding)
        zero_allele = torch.zeros_like(allele_embedding)
        out_ctx = self.forward(
            node_features_seq, zero_allele, bin_boundaries=bin_boundaries
        )
        _, _, context_laplacians = out_ctx

        # Delta: allele-specific perturbation
        delta_laplacians = [
            full_L - ctx_L
            for full_L, ctx_L in zip(full_laplacians, context_laplacians)
        ]

        return context_laplacians, full_laplacians, delta_laplacians

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
        counts["initial_maps"] = self.initial_maps.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


def create_tsgnn_from_config(config: dict, edge_index: torch.Tensor) -> TSGNN:
    """Factory function to create TS-GNN from a config dict."""
    # num_nodes = number of genes (N), distinct from input_dim (feature size per gene)
    num_nodes = config.get("data", {}).get("n_genes", config["model"]["input_dim"])
    model_cfg = config.get("model", {})
    return TSGNN(
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
        stalk_dim=model_cfg["stalk_dim"],
        input_dim=model_cfg["input_dim"],
        esm_dim=model_cfg["esm_dim"],
        conditioning_dim=model_cfg["conditioning_dim"],
        edge_index=edge_index,
        num_diffusion_steps=model_cfg["num_diffusion_steps"],
        use_allele_conditioning=model_cfg.get("use_allele_conditioning", True),
        use_vectorized=model_cfg.get("use_vectorized", True),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", False),
        use_normalized_laplacian=model_cfg.get("use_normalized_laplacian", True),
        use_implicit_diffusion=model_cfg.get("use_implicit_diffusion", True),
        predict_delta=model_cfg.get("predict_delta", True),
        use_neural_ode=model_cfg.get("use_neural_ode", True),
        ode_hidden_dim=model_cfg.get("ode_hidden_dim", 256),
        ode_solver=model_cfg.get("ode_solver", "rk4"),
        edge_weight_hidden=model_cfg.get("edge_weight_hidden", 128),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test with synthetic data
    N, E, d, input_dim, K = 50, 200, 4, 50, 10
    esm_dim = 1280

    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=input_dim,
        esm_dim=esm_dim,
        conditioning_dim=128,
        edge_index=edge_index,
        num_diffusion_steps=3,
    )

    # Synthetic temporal data
    node_features_seq = torch.randn(K, N, input_dim)
    allele_emb = torch.randn(esm_dim)

    predictions, maps_traj, laplacians, stalks_traj = model(node_features_seq, allele_emb)

    print(f"Model created: {N} nodes, {E} edges, d={d}, K={K}")
    print(f"Predictions: {len(predictions)} steps, each {predictions[0].shape}")
    print(f"Maps trajectory: {len(maps_traj)} steps, each {maps_traj[0].shape}")
    print(f"Laplacians: {len(laplacians)} steps, each {laplacians[0].shape}")

    params = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test decomposition
    ctx_L, full_L, delta_L = model.decompose_rewiring(node_features_seq, allele_emb)
    print(f"\nRewiring decomposition:")
    print(f"  Context Laplacian norm: {ctx_L[0].norm():.4f}")
    print(f"  Full Laplacian norm: {full_L[0].norm():.4f}")
    print(f"  Delta (allele-specific) norm: {delta_L[0].norm():.4f}")
