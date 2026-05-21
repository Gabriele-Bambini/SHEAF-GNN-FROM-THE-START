"""
Rewiring Trajectory Visualization.

Phase 5.1: Animate the GRN evolving over pseudotime for each allele.
- Edge weight changes as color-coded arrows
- Allele-specific edges highlighted
- Transient bottleneck state identification
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_rewiring_trajectory(
    laplacians: List[torch.Tensor],
    edge_index: torch.Tensor,
    stalk_dim: int,
    gene_names: Optional[List[str]] = None,
    allele: str = "unknown",
    output_dir: str = "figures",
    top_k_edges: int = 30,
):
    """
    Plot GRN rewiring trajectory over pseudotime.

    Creates a multi-panel figure showing edge weight evolution.

    Args:
        laplacians: List of K sheaf Laplacians (N*d, N*d).
        edge_index: (2, E) edge indices.
        stalk_dim: Stalk dimension d.
        gene_names: Optional gene name labels.
        allele: Allele name for title.
        output_dir: Output directory.
        top_k_edges: Number of top-changing edges to highlight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    K = len(laplacians)
    d = stalk_dim
    Nd = laplacians[0].shape[0]
    N = Nd // d

    # Extract edge weights over time (off-diagonal block norms)
    E = edge_index.shape[1]
    edge_weights = np.zeros((K, E))
    for t in range(K):
        L = laplacians[t].detach().cpu()
        for e in range(E):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            block = L[u * d:(u + 1) * d, v * d:(v + 1) * d]
            edge_weights[t, e] = block.norm().item()

    # Find top-changing edges
    edge_change = np.abs(edge_weights[-1] - edge_weights[0])
    top_indices = np.argsort(edge_change)[-top_k_edges:]

    # Plot: edge weight heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Heatmap of top-changing edges over time
    ax = axes[0]
    im = ax.imshow(
        edge_weights[:, top_indices].T,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("Edge index (top changing)")
    ax.set_title(f"Edge weight evolution - {allele}")
    plt.colorbar(im, ax=ax, label="Edge weight (block norm)")

    # Panel B: Line plot of top-5 edges
    ax = axes[1]
    for idx in top_indices[-5:]:
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        label = f"{gene_names[u]}->{gene_names[v]}" if gene_names else f"{u}->{v}"
        ax.plot(range(K), edge_weights[:, idx], marker="o", label=label, linewidth=2)
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("Edge weight")
    ax.set_title(f"Top rewired edges - {allele}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"rewiring_trajectory_{allele}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path / f"rewiring_trajectory_{allele}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rewiring trajectory plot for {allele}")


def plot_differential_rewiring(
    laplacians_a: List[torch.Tensor],
    laplacians_b: List[torch.Tensor],
    edge_index: torch.Tensor,
    stalk_dim: int,
    allele_a: str = "R175H",
    allele_b: str = "R273H",
    gene_names: Optional[List[str]] = None,
    output_dir: str = "figures",
):
    """
    Plot differential rewiring between two alleles.

    Highlights edges that differ most between the two alleles' trajectories.
    """
    import matplotlib.pyplot as plt

    K = min(len(laplacians_a), len(laplacians_b))
    d = stalk_dim
    E = edge_index.shape[1]
    Nd = laplacians_a[0].shape[0]
    N = Nd // d

    # Compute differential edge weights
    diff_weights = np.zeros((K, E))
    for t in range(K):
        La = laplacians_a[t].detach().cpu()
        Lb = laplacians_b[t].detach().cpu()
        for e in range(E):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            block_a = La[u * d:(u + 1) * d, v * d:(v + 1) * d]
            block_b = Lb[u * d:(u + 1) * d, v * d:(v + 1) * d]
            diff_weights[t, e] = (block_a - block_b).norm().item()

    # Top differential edges
    mean_diff = diff_weights.mean(axis=0)
    top_idx = np.argsort(mean_diff)[-20:]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        diff_weights[:, top_idx].T,
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
    )
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("Edge index (top differential)")
    ax.set_title(f"Differential rewiring: {allele_a} vs {allele_b}")
    plt.colorbar(im, ax=ax, label="||L_a - L_b|| (block norm)")

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"diff_rewiring_{allele_a}_vs_{allele_b}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved differential rewiring plot: {allele_a} vs {allele_b}")


def identify_bottleneck_timepoints(
    laplacians: List[torch.Tensor],
    edge_index: torch.Tensor,
    stalk_dim: int,
) -> List[Dict]:
    """
    Identify time points with maximal betweenness centrality change.

    These represent transient bottleneck states where the network topology
    undergoes rapid restructuring — potential therapeutic windows.

    Returns:
        List of dicts with time_index, centrality_change, top_nodes.
    """
    import networkx as nx

    K = len(laplacians)
    d = stalk_dim
    Nd = laplacians[0].shape[0]
    N = Nd // d

    centralities = []
    for t in range(K):
        L = laplacians[t].detach().cpu()

        # Build weighted graph from Laplacian off-diagonal blocks
        G = nx.DiGraph()
        for e in range(edge_index.shape[1]):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            block = L[u * d:(u + 1) * d, v * d:(v + 1) * d]
            weight = block.norm().item()
            if weight > 1e-6:
                G.add_edge(u, v, weight=weight)

        if G.number_of_edges() > 0:
            bc = nx.betweenness_centrality(G, weight="weight")
            centralities.append(bc)
        else:
            centralities.append({i: 0.0 for i in range(N)})

    # Compute centrality changes between consecutive steps
    bottlenecks = []
    for t in range(1, K):
        changes = {}
        for node in range(N):
            prev = centralities[t - 1].get(node, 0.0)
            curr = centralities[t].get(node, 0.0)
            changes[node] = abs(curr - prev)

        total_change = sum(changes.values())
        top_nodes = sorted(changes.keys(), key=lambda n: changes[n], reverse=True)[:5]

        bottlenecks.append({
            "time_index": t,
            "centrality_change": total_change,
            "top_nodes": top_nodes,
            "top_changes": [changes[n] for n in top_nodes],
        })

    # Sort by magnitude of change
    bottlenecks.sort(key=lambda x: x["centrality_change"], reverse=True)
    return bottlenecks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    K, N, d, E = 10, 20, 4, 50
    Nd = N * d
    edge_index = torch.randint(0, N, (2, E))
    laplacians = [torch.randn(Nd, Nd) for _ in range(K)]

    bottlenecks = identify_bottleneck_timepoints(laplacians, edge_index, d)
    print(f"Top bottleneck: t={bottlenecks[0]['time_index']}, "
          f"change={bottlenecks[0]['centrality_change']:.4f}")

    print("Visualization module ready. Call plot functions with matplotlib available.")
