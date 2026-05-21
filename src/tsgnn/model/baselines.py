"""
Baseline Models for Fair Comparison with TS-GNN.

Phase 2E: Implements 5 baselines that share identical data preprocessing,
temporal binning (K=10), evaluation metrics, and train/val/test splits.

Baselines:
1. CellOracle wrapper   — in silico TP53 perturbation GRN inference
2. Dictys wrapper       — time-resolved GRN from dynamics
3. EvolveGCN            — GCNConv + GRU-evolved parameters (no sheaf)
4. TS-GNN no-allele     — Full architecture with FiLM disabled (ablation)
5. Temporal GAT         — GATConv + GRU + FiLM (no sheaf geometry)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CellOracle Wrapper
# ---------------------------------------------------------------------------

class CellOracleBaseline:
    """
    Wrapper around CellOracle for in silico TP53 perturbation.

    CellOracle (Kamimoto et al., Nature 2023) infers GRNs from scRNA-seq
    and simulates gene expression shifts upon TF perturbation.

    This is an external tool; the wrapper handles:
    - Running CellOracle's pipeline on our preprocessed AnnData
    - Extracting per-condition GRN edges for comparison
    - Converting outputs to the same format as TS-GNN predictions
    """

    def __init__(self, adata, gene_list: List[str]):
        self.adata = adata
        self.gene_list = gene_list
        self.grn_edges: Optional[Dict] = None

    def run(self, tp53_perturbation: float = 0.0) -> Dict:
        """
        Run CellOracle in silico perturbation.

        Args:
            tp53_perturbation: Expression shift for TP53 (0 = knockout).

        Returns:
            Dict with 'edges' (source, target, weight) and 'expression_shift'.
        """
        try:
            import celloracle as co

            logger.info("Running CellOracle GRN inference...")

            # Initialize Oracle object
            oracle = co.Oracle()
            oracle.import_anndata_as_raw_count(
                adata=self.adata,
                cluster_column_name="leiden",
                embedding_name="X_umap",
            )

            # Load base GRN (from JASPAR motif scan or user-provided)
            # CellOracle requires a base GRN as prior
            oracle.import_TF_data(TF_info_matrix=None)  # Use default

            # Fit GRN per cluster
            oracle.perform_PCA()
            oracle.knn_imputation()

            links = oracle.get_links(
                cluster_name_for_GRN_unit="leiden",
                alpha=10,
                verbose_level=0,
            )

            # Simulate TP53 perturbation
            oracle.simulate_shift(
                perturb_condition={"TP53": tp53_perturbation},
                n_propagation=3,
            )

            # Extract results
            shift = oracle.adata.layers.get("simulated_count", None)
            edges = links.filtered_links if hasattr(links, "filtered_links") else {}

            result = {
                "edges": edges,
                "expression_shift": shift,
                "method": "CellOracle",
            }
            self.grn_edges = result
            logger.info("CellOracle perturbation complete.")
            return result

        except ImportError:
            logger.warning(
                "CellOracle not installed. Install with: pip install celloracle\n"
                "Returning placeholder results."
            )
            return self._placeholder_results()

        except Exception as e:
            logger.warning(f"CellOracle failed: {e}. Returning placeholder results.")
            return self._placeholder_results()

    def _placeholder_results(self) -> Dict:
        N = len(self.gene_list)
        return {
            "edges": {},
            "expression_shift": torch.zeros(N),
            "method": "CellOracle (placeholder)",
        }


# ---------------------------------------------------------------------------
# 2. Dictys Wrapper
# ---------------------------------------------------------------------------

class DictysBaseline:
    """
    Wrapper around Dictys for time-resolved GRN inference.

    Dictys (Wang et al., Nature Methods 2023) infers dynamic GRNs from
    multi-omic data using neural ODEs on temporal scRNA-seq.

    This wrapper:
    - Runs Dictys on the same temporal bins as TS-GNN
    - Extracts time-resolved GRN edge weights
    - Converts to comparable format
    """

    def __init__(self, adata, gene_list: List[str], K: int = 10):
        self.adata = adata
        self.gene_list = gene_list
        self.K = K

    def run(self) -> Dict:
        """
        Run Dictys dynamic GRN inference.

        Returns:
            Dict with 'temporal_edges' (K x N x N sparse) and metadata.
        """
        try:
            import dictys

            logger.info("Running Dictys dynamic GRN inference...")

            # Dictys typically requires:
            # 1. Preprocessed expression matrix
            # 2. Chromatin accessibility (optional for RNA-only mode)
            # 3. TF binding motifs

            # Run in RNA-only mode
            result = dictys.net.network(
                expression=self.adata,
                gene_names=self.gene_list,
            )

            return {
                "temporal_edges": result,
                "method": "Dictys",
            }

        except ImportError:
            logger.warning(
                "Dictys not installed. Install with: pip install dictys\n"
                "Returning placeholder results."
            )
            return self._placeholder_results()

        except Exception as e:
            logger.warning(f"Dictys failed: {e}. Returning placeholder results.")
            return self._placeholder_results()

    def _placeholder_results(self) -> Dict:
        N = len(self.gene_list)
        return {
            "temporal_edges": [torch.zeros(N, N) for _ in range(self.K)],
            "method": "Dictys (placeholder)",
        }


# ---------------------------------------------------------------------------
# 3. EvolveGCN (no sheaf geometry)
# ---------------------------------------------------------------------------

class EvolveGCN(nn.Module):
    """
    EvolveGCN baseline: GCNConv with GRU-evolved parameters.

    Replaces sheaf diffusion with standard GCN message passing.
    Uses a GRU to evolve GCN weight matrices over pseudotime,
    giving temporal adaptivity without sheaf geometry.

    Architecture per time step t:
        1. Project input to hidden: h = proj(X(t))
        2. Evolve GCN weights via GRU: W(t+1) = GRU(W(t), context)
        3. GCN message passing: h = GCN(h, A, W(t+1))  x num_layers
        4. Project back: X_hat(t+1) = proj_out(h)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.register_buffer("edge_index", edge_index)

        # Input/output projections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # GCN layers (manual implementation to avoid hard PyG dependency at import)
        self.gcn_weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.gcn_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim))
            for _ in range(num_layers)
        ])

        # GRU to evolve GCN weights over time
        weight_dim = hidden_dim * hidden_dim
        self.weight_gru = nn.GRUCell(
            input_size=weight_dim,
            hidden_size=weight_dim,
        )

        self.activation = nn.ReLU()
        self._init_parameters()

    def _init_parameters(self):
        for w in self.gcn_weights:
            nn.init.xavier_uniform_(w)

    def _gcn_conv(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Simple GCN convolution: D^{-1/2} A D^{-1/2} X W + b."""
        N = x.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        # Compute degree for normalization
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=x.device))
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Normalize: D^{-1/2} A D^{-1/2}
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[tgt]  # (E,)

        # Message passing
        messages = x[src] * norm.unsqueeze(-1)  # (E, hidden)
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(messages), messages)

        # Transform
        out = agg @ weight + bias
        return out

    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor = None,
        edge_weights_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass matching TS-GNN interface.

        Returns:
            predictions, empty maps trajectory, empty Laplacians.
        """
        K = node_features_seq.shape[0]
        predictions = []

        # Initialize GRU hidden state from first GCN weight
        weight_hidden = self.gcn_weights[0].reshape(-1).unsqueeze(0)  # (1, h*h)

        for t in range(K):
            x = self.input_proj(node_features_seq[t])  # (N, hidden)

            # Evolve weights
            weight_hidden = self.weight_gru(weight_hidden, weight_hidden)
            evolved_weight = weight_hidden.reshape(self.hidden_dim, self.hidden_dim)

            # GCN layers
            for layer_idx in range(self.num_layers):
                w = evolved_weight if layer_idx == 0 else self.gcn_weights[layer_idx]
                x = self._gcn_conv(x, self.edge_index, w, self.gcn_biases[layer_idx])
                if layer_idx < self.num_layers - 1:
                    x = self.activation(x)

            pred = self.output_proj(x)
            predictions.append(pred)

        # Return empty lists for maps/laplacians (interface compatibility)
        empty = [torch.tensor(0.0)] * K
        return predictions, empty, empty

    def count_parameters(self) -> Dict[str, int]:
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                counts[name] = n
        counts["gcn_weights"] = sum(p.numel() for p in self.gcn_weights)
        counts["gcn_biases"] = sum(p.numel() for p in self.gcn_biases)
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ---------------------------------------------------------------------------
# 4. TS-GNN without allele conditioning (ablation)
# ---------------------------------------------------------------------------

def create_tsgnn_no_allele(
    num_nodes: int,
    num_edges: int,
    stalk_dim: int,
    input_dim: int,
    edge_index: torch.Tensor,
    num_diffusion_steps: int = 3,
):
    """
    Create TS-GNN with FiLM disabled (gamma=1, beta=0 always).

    This is the primary ablation baseline: same sheaf geometry + GRU
    temporal evolution, but no allele-specific modulation.
    """
    from .tsgnn import TSGNN

    return TSGNN(
        num_nodes=num_nodes,
        num_edges=num_edges,
        stalk_dim=stalk_dim,
        input_dim=input_dim,
        esm_dim=1280,  # unused but required for init
        conditioning_dim=128,
        edge_index=edge_index,
        num_diffusion_steps=num_diffusion_steps,
        use_allele_conditioning=False,  # ZeroFiLM: identity transform
    )


# ---------------------------------------------------------------------------
# 5. Temporal GAT (no sheaf geometry)
# ---------------------------------------------------------------------------

class TemporalGAT(nn.Module):
    """
    Temporal Graph Attention Network baseline.

    Replaces sheaf diffusion with multi-head GAT attention.
    Retains GRU temporal evolution and FiLM allele conditioning,
    but uses standard attention instead of sheaf geometry.

    Architecture per time step t:
        1. Project input to hidden: h = proj(X(t))
        2. Evolve attention params via GRU
        3. GAT message passing with multi-head attention
        4. FiLM conditioning from allele embedding
        5. Project back: X_hat(t+1) = proj_out(h)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
        esm_dim: int = 1280,
        conditioning_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        use_allele_conditioning: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.register_buffer("edge_index", edge_index)

        # Input/output projections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Multi-head GAT attention layers
        head_dim = hidden_dim // num_heads
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                GATLayer(hidden_dim, head_dim, num_heads)
            )

        # GRU for temporal evolution of attention parameters
        attn_param_dim = num_heads * (2 * head_dim + 1)  # per-layer attention params
        self.attn_gru = nn.GRUCell(
            input_size=attn_param_dim,
            hidden_size=attn_param_dim,
        )
        self.attn_hidden_proj = nn.Linear(attn_param_dim, attn_param_dim)

        self.use_allele_conditioning = use_allele_conditioning
        if use_allele_conditioning:
            self.allele_proj = nn.Linear(hidden_dim + esm_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
        edge_weights_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass matching TS-GNN interface.

        Returns:
            predictions, empty maps trajectory, empty Laplacians.
        """
        K = node_features_seq.shape[0]
        predictions = []

        # Initialize GRU hidden for attention parameters
        attn_param_dim = self.attn_gru.hidden_size
        attn_hidden = torch.zeros(1, attn_param_dim, device=node_features_seq.device)

        for t in range(K):
            x = self.input_proj(node_features_seq[t])  # (N, hidden)

            # Evolve attention parameters via GRU
            attn_hidden = self.attn_gru(attn_hidden, attn_hidden)

            # GAT message passing
            for layer in self.attn_layers:
                x = layer(x, self.edge_index)
                x = self.layer_norm(x)

            # Allele conditioning via concatenation
            if self.use_allele_conditioning:
                N = x.shape[0]
                emb_expanded = allele_embedding.reshape(-1).unsqueeze(0).expand(N, -1)
                x = torch.cat([x, emb_expanded], dim=-1)
                x = torch.relu(self.allele_proj(x))

            pred = self.output_proj(x)
            predictions.append(pred)

        empty = [torch.tensor(0.0)] * K
        return predictions, empty, empty

    def count_parameters(self) -> Dict[str, int]:
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                counts[name] = n
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


class GATLayer(nn.Module):
    """Single multi-head GAT attention layer."""

    def __init__(self, in_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.W = nn.Linear(in_dim, num_heads * head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, head_dim))
        self.a_tgt = nn.Parameter(torch.empty(num_heads, head_dim))
        self.proj_out = nn.Linear(num_heads * head_dim, in_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_tgt.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Multi-head GAT attention.

        Args:
            x: (N, in_dim) node features.
            edge_index: (2, E) edges.

        Returns:
            Updated features (N, in_dim).
        """
        N = x.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        # Linear transform and reshape to heads
        h = self.W(x).reshape(N, self.num_heads, self.head_dim)  # (N, H, D)

        # Attention scores
        e_src = (h[src] * self.a_src).sum(dim=-1)  # (E, H)
        e_tgt = (h[tgt] * self.a_tgt).sum(dim=-1)  # (E, H)
        e = self.leaky_relu(e_src + e_tgt)  # (E, H)

        # Softmax per target node
        e_max = torch.zeros(N, self.num_heads, device=x.device)
        e_max.scatter_reduce_(0, tgt.unsqueeze(-1).expand_as(e), e, reduce="amax")
        e = torch.exp(e - e_max[tgt])

        e_sum = torch.zeros(N, self.num_heads, device=x.device)
        e_sum.scatter_add_(0, tgt.unsqueeze(-1).expand_as(e), e)
        alpha = e / (e_sum[tgt] + 1e-8)  # (E, H)

        # Weighted message passing
        messages = h[src] * alpha.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            tgt.unsqueeze(-1).unsqueeze(-1).expand_as(messages),
            messages,
        )

        # Concatenate heads and project
        out = out.reshape(N, -1)  # (N, H*D)
        out = self.proj_out(out)  # (N, in_dim)

        # Residual connection
        return x + out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. MLP (no-graph control)
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """Node-wise MLP, ignores graph and temporal structure.
    Controls for the statistical power contributed by the neural net alone.
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int,
                 edge_index: Optional[torch.Tensor] = None, num_layers: int = 3, **_):
        super().__init__()
        self.num_nodes = num_nodes
        layers: list = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.ReLU()]
            d_in = hidden_dim
        layers += [nn.Linear(d_in, input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, node_features_seq, allele_embedding=None, edge_weights_seq=None):
        if node_features_seq.dim() == 2:
            return self.net(node_features_seq)
        K = node_features_seq.shape[0]
        preds = [self.net(node_features_seq[t]) for t in range(K)]
        empty = [torch.tensor(0.0)] * K
        return preds, empty, empty


# ---------------------------------------------------------------------------
# 7. Static GCN (no temporal evolution)
# ---------------------------------------------------------------------------

class GCNStaticBaseline(nn.Module):
    """Standard GCN repeated identically at every time step.
    Controls for what sheaf + temporal evolution buy over static message passing.
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int,
                 edge_index: torch.Tensor, num_layers: int = 2, **_):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("edge_index", edge_index)
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim))
                                          for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim))
                                         for _ in range(num_layers)])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def _gcn(self, x, w, b):
        src, tgt = self.edge_index[0], self.edge_index[1]
        N = x.shape[0]
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=x.device))
        deg = deg.clamp(min=1).pow(-0.5)
        norm = deg[src] * deg[tgt]
        msg = x[src] * norm.unsqueeze(-1)
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        return agg @ w + b

    def _step(self, x):
        h = torch.relu(self.input_proj(x))
        for i in range(self.num_layers):
            h = self._gcn(h, self.weights[i], self.biases[i])
            if i < self.num_layers - 1:
                h = torch.relu(h)
        return self.output_proj(h)

    def forward(self, node_features_seq, allele_embedding=None, edge_weights_seq=None):
        if node_features_seq.dim() == 2:
            return self._step(node_features_seq)
        K = node_features_seq.shape[0]
        preds = [self._step(node_features_seq[t]) for t in range(K)]
        empty = [torch.tensor(0.0)] * K
        return preds, empty, empty


# ---------------------------------------------------------------------------
# 8. DySAT (structural + temporal self-attention)
# ---------------------------------------------------------------------------

class DySATBaseline(nn.Module):
    """DySAT (Sankar et al. 2020): structural attention over neighbors at each
    timestep, then temporal multi-head self-attention across the K snapshots.
    Compact implementation sufficient for benchmarking — not the reference impl.
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int,
                 edge_index: torch.Tensor, num_heads: int = 4, **_):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads)
        self.temp_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, node_features_seq, allele_embedding=None, edge_weights_seq=None):
        K, N, _ = node_features_seq.shape
        # Structural attention per timestep
        H = []
        for t in range(K):
            h = torch.relu(self.input_proj(node_features_seq[t]))
            h = self.gat(h, self.edge_index)
            H.append(h)
        seq = torch.stack(H, dim=1)                          # (N, K, hidden)
        # Temporal self-attention
        out, _ = self.temp_attn(seq, seq, seq)               # (N, K, hidden)
        preds = [self.output_proj(out[:, t, :]) for t in range(K)]
        empty = [torch.tensor(0.0)] * K
        return preds, empty, empty


BASELINE_REGISTRY = {
    "celloracle": CellOracleBaseline,
    "dictys": DictysBaseline,
    "evolvegcn": EvolveGCN,
    "tsgnn_no_allele": create_tsgnn_no_allele,
    "temporal_gat": TemporalGAT,
    "mlp": MLPBaseline,
    "gcn_static": GCNStaticBaseline,
    "dysat": DySATBaseline,
}


def create_baseline(name: str, **kwargs):
    """Factory function to create a baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    N, E, hidden, input_dim, K = 50, 200, 16, 50, 10
    edge_index = torch.randint(0, N, (2, E))
    node_features_seq = torch.randn(K, N, input_dim)
    allele_emb = torch.randn(1280)

    # Test EvolveGCN
    print("=== EvolveGCN ===")
    model = EvolveGCN(
        num_nodes=N, input_dim=input_dim, hidden_dim=hidden, edge_index=edge_index
    )
    preds, _, _ = model(node_features_seq)
    print(f"Predictions: {len(preds)} steps, each {preds[0].shape}")
    print(f"Parameters: {model.count_parameters()}")

    # Test TS-GNN no-allele
    print("\n=== TS-GNN (no allele) ===")
    model_no_allele = create_tsgnn_no_allele(
        num_nodes=N, num_edges=E, stalk_dim=4, input_dim=input_dim, edge_index=edge_index
    )
    preds, maps, laps = model_no_allele(node_features_seq, allele_emb)
    print(f"Predictions: {len(preds)} steps, each {preds[0].shape}")
    print(f"Parameters: {model_no_allele.count_parameters()}")

    # Test Temporal GAT
    print("\n=== Temporal GAT ===")
    gat = TemporalGAT(
        num_nodes=N, input_dim=input_dim, hidden_dim=hidden, edge_index=edge_index
    )
    preds, _, _ = gat(node_features_seq, allele_emb)
    print(f"Predictions: {len(preds)} steps, each {preds[0].shape}")
    print(f"Parameters: {gat.count_parameters()}")
