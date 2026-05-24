"""
Full TS-GNN (Temporal Sheaf Graph Neural Network) Architecture.

Lie-Sheaf allele model (post-mortem 2026-05-18):
Architecture flow at each time step t:
1. Project input features to stalk space: x = proj(X(t))
2. Evolve restriction maps via GRU: F_base(t+1) = GRU(F_base(t), context)
3. Project to O(d): F_base_orth = QR(F_base)
4. Apply allele-class rotation: F_allele = F_base_orth @ R_class  (SO(4), 18 params)
5. Compute connection Laplacian: L_F = sheaf_laplacian(F_allele)
6. Sheaf diffusion (multiple steps): x = diffuse(x, L_F)
7. Project back to expression space: X_hat(t+1) = proj_out(x)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .sheaf import SheafDiffusionLayer
from .sheaf_vectorized import NormalizedVectorizedSheafDiffusion
from .sheaf_sparse import SparseVectorizedSheafDiffusion
from .sheaf_orthogonal import OrthogonalSheafDiffusion, project_maps_bundle
from .diffusion_implicit import ImplicitOrthogonalSheafDiffusion
from .temporal import TemporalRestrictionEvolution
from .allele_so4 import AlleleSO4Embedding

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


class TSGNN(nn.Module):
    """
    Temporal Sheaf Graph Neural Network for allele-conditioned GRN rewiring.

    Lie-Sheaf allele model: TP53 allele classes are SO(4) rotations on restriction maps.
    18-parameter AlleleSO4Embedding replaces FiLM+SheafPerturbation+ODE.

    Args:
        num_nodes: N, number of genes/TFs.
        num_edges: E, number of regulatory edges.
        stalk_dim: d, sheaf stalk dimension.
        input_dim: Dimension of input gene features.
        n_allele_classes: Number of allele classes (3: WT/structural/dna_contact).
        allele_init_scale: Init std for SO(4) parameters. 0.1 = near identity.
        edge_index: (2, E) directed edge tensor.
        num_diffusion_steps: Sheaf diffusion iterations per time step.
        use_orthogonal_maps: If True, use implicit Crank-Nicolson diffusion.
        use_implicit_diffusion: If True, use CN implicit solver.
        use_gradient_checkpointing: If True, trade memory for compute.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        input_dim: int,
        edge_index: torch.Tensor,
        n_allele_classes: int = 3,
        allele_init_scale: float = 0.1,
        num_diffusion_steps: int = 3,
        use_vectorized: bool = True,
        use_gradient_checkpointing: bool = False,
        use_normalized_laplacian: bool = True,
        use_orthogonal_maps: bool = True,
        use_implicit_diffusion: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.d = stalk_dim
        self.input_dim = input_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.use_vectorized = use_vectorized
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_normalized_laplacian = use_normalized_laplacian
        self.use_orthogonal_maps = use_orthogonal_maps
        self.use_implicit_diffusion = use_implicit_diffusion

        self.register_buffer("edge_index", edge_index)

        # Input projection: gene expression -> stalk space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, stalk_dim),
        )

        # Sheaf diffusion layer (A1+A2: orthogonal maps + Crank-Nicolson implicit)
        if use_implicit_diffusion:
            self.sheaf_layer = ImplicitOrthogonalSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        elif use_orthogonal_maps:
            self.sheaf_layer = OrthogonalSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        elif use_vectorized and use_normalized_laplacian:
            self.sheaf_layer = NormalizedVectorizedSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        elif use_vectorized:
            self.sheaf_layer = SparseVectorizedSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        else:
            self.sheaf_layer = SheafDiffusionLayer(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )

        # Temporal evolution of restriction maps (GRU — ODE removed, post-mortem 2026-05-18)
        self.temporal_evolution = TemporalRestrictionEvolution(stalk_dim=stalk_dim)

        # Allele conditioning: SO(4) rotation per class (18 parameters total)
        # Replaces FiLM + SheafPerturbation. F_allele = F_base @ R_class
        self.allele_so4 = AlleleSO4Embedding(
            n_allele_classes=n_allele_classes,
            stalk_dim=stalk_dim,
            init_scale=allele_init_scale,
        )

        # Stalk Interpreter (Interpretability Layer)
        self.interpreter = StalkInterpreter(stalk_dim)

        # Output projection: stalk space -> expression prediction
        self.output_proj = nn.Sequential(
            nn.Linear(stalk_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, input_dim),
        )

        # Direction gates: per-timestep learnable diffusion direction.
        # Initialized to +0.55 → tanh(0.55) ≈ 0.5 (half-strength smoothing).
        # Can learn negative values for early bins where Laplacian diffusion
        # opposes actual ΔX (reverse diffusion / sharpening).
        # tanh bounds to [-1, +1], multiplied by base tau.
        max_K = 20  # max temporal bins supported
        self.direction_logits = nn.Parameter(torch.full((max_K,), 0.55))

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

    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_class: torch.LongTensor,
        edge_weights_seq: Optional[torch.Tensor] = None,
        bin_boundaries: Optional[torch.Tensor] = None,
        return_stalks: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass over the temporal graph sequence.

        Args:
            node_features_seq: (K, N, input_dim) temporal node features.
            allele_class: scalar LongTensor in {0, 1, 2}
                          (0=WT, 1=structural, 2=dna_contact).
            edge_weights_seq: Optional (K, E) temporal edge weights (unused).

        Returns:
            predictions: List of K tensors, each (N, input_dim).
            restriction_maps_trajectory: List of K tensors, each (E, 2, d, d).
            sheaf_laplacians: List of K tensors, each (N*d, N*d).
        """
        K = node_features_seq.shape[0]
        predictions = []
        maps_trajectory = []
        laplacians = []
        stalks_trajectory = []

        # Pre-compute allele rotation once (shared across all timesteps)
        R_allele = self.allele_so4(allele_class)  # (d, d) in SO(d)

        current_maps = self.initial_maps.clone()

        def _step(current_maps_inner, node_feat_t, time_idx=0):
            # 1. Project input to stalk space
            x_latent = self.input_proj(node_feat_t)  # (N, d)

            # 2. Evolve restriction maps via GRU
            new_maps = self.temporal_evolution(
                current_maps_inner, x_latent, self.edge_index, None
            )

            # 3. Project to O(d) base maps
            base_orth = project_maps_bundle(new_maps)  # (E, 2, d, d)

            # 4. Apply allele-class rotation: F_allele = F_base @ R_class
            cmaps = torch.einsum('esij,jk->esik', base_orth, R_allele)  # (E, 2, d, d)

            # 5. Build sheaf Laplacian
            L_F = self.sheaf_layer.compute_connection_laplacian(cmaps)

            # 6. Sheaf diffusion (direction-aware)
            dir_idx = min(time_idx, len(self.direction_logits) - 1)
            direction = torch.tanh(self.direction_logits[dir_idx])
            x = x_latent
            for _ in range(self.num_diffusion_steps):
                if hasattr(self.sheaf_layer, 'tau'):
                    tau_directed = self.sheaf_layer.tau * direction
                    x = self.sheaf_layer.diffuse(x, L_F, tau_override=tau_directed)
                else:
                    x = self.sheaf_layer.diffuse(x, L_F)

            # 7. Output
            x_interpreted = self.interpreter(x)
            x_pred = self.output_proj(x_interpreted)
            return x_pred, cmaps, L_F, x_interpreted

        for t in range(K):
            if self.use_gradient_checkpointing and self.training:
                x_pred, current_maps, L_F, stalks_t = grad_ckpt(
                    _step, current_maps, node_features_seq[t], t,
                    use_reentrant=False,
                )
            else:
                x_pred, current_maps, L_F, stalks_t = _step(
                    current_maps, node_features_seq[t], t
                )

            maps_trajectory.append(current_maps.detach().clone())
            laplacians.append(L_F)
            predictions.append(x_pred)
            stalks_trajectory.append(stalks_t)

        if return_stalks:
            return predictions, maps_trajectory, laplacians, stalks_trajectory
        return predictions, maps_trajectory, laplacians

    def decompose_rewiring(
        self,
        node_features_seq: torch.Tensor,
        allele_class: torch.LongTensor,
        wt_class: int = 0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Decompose rewiring into WT + allele-specific components.

        Runs forward twice (allele_class and WT class=0).

        Returns:
            wt_laplacians: List of K Laplacians from WT model.
            allele_laplacians: List of K full Laplacians.
            delta_laplacians: List of K (allele - WT) perturbation Laplacians.
        """
        dev = node_features_seq.device
        wt_tensor = torch.tensor(wt_class, dtype=torch.long, device=dev)

        _, _, full_laplacians = self.forward(node_features_seq, allele_class)
        _, _, wt_laplacians = self.forward(node_features_seq, wt_tensor)

        delta_laplacians = [
            full_L - wt_L
            for full_L, wt_L in zip(full_laplacians, wt_laplacians)
        ]
        return wt_laplacians, full_laplacians, delta_laplacians

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
    num_nodes = config.get("data", {}).get("n_genes", 500)
    model_cfg = config.get("model", {})
    return TSGNN(
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
        stalk_dim=model_cfg.get("stalk_dim", 4),
        input_dim=model_cfg.get("input_dim", 1),
        n_allele_classes=config.get("allele_classes", {}).get("n_classes", 3),
        allele_init_scale=model_cfg.get("allele_init_scale", 0.1),
        edge_index=edge_index,
        num_diffusion_steps=model_cfg.get("num_diffusion_steps", 3),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", False),
        use_normalized_laplacian=model_cfg.get("use_normalized_laplacian", True),
        use_orthogonal_maps=model_cfg.get("use_orthogonal_maps", True),
        use_implicit_diffusion=model_cfg.get("use_implicit_diffusion", True),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    N, E, d, input_dim, K = 50, 200, 4, 1, 10
    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=d, input_dim=input_dim,
        n_allele_classes=3, edge_index=edge_index,
    )

    node_features_seq = torch.randn(K, N, input_dim)
    predictions, maps_traj, laplacians = model(node_features_seq, torch.tensor(1))

    print(f"Model: {N} nodes, {E} edges, d={d}, K={K}")
    print(f"Predictions: {len(predictions)} steps, each {predictions[0].shape}")
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,} total, {params['allele_so4']} SO(4)")
