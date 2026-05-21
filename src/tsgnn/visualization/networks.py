"""
GRN and Allele Embedding Space Visualization.

Phase 5.3:
- t-SNE/UMAP of ESM-2 allele embeddings colored by structural class
- GRN visualization with edge coloring by regulatory mode
- Allele-specific subnetwork highlighting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_allele_embedding_space(
    esm_embeddings: Dict[str, torch.Tensor],
    structural_classes: Optional[Dict[str, str]] = None,
    method: str = "tsne",
    output_dir: str = "figures",
):
    """
    Visualize ESM-2 allele embeddings in 2D.

    Colors by structural class (contact vs. conformational) to show
    that structurally similar alleles cluster together.

    Args:
        esm_embeddings: Dict mapping allele name -> (esm_dim,) tensor.
        structural_classes: Dict mapping allele -> "contact" or "conformational".
        method: "tsne" or "umap".
        output_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    if structural_classes is None:
        structural_classes = {
            "WT": "wild_type",
            "R273H": "contact",
            "R248W": "contact",
            "R175H": "conformational",
            "G245S": "conformational",
            "R282W": "conformational",
            "Y220C": "conformational",
        }

    allele_names = list(esm_embeddings.keys())
    X = torch.stack([esm_embeddings[a].detach().cpu() for a in allele_names]).numpy()

    # Dimensionality reduction
    if X.shape[0] < 3:
        logger.warning("Too few alleles for meaningful embedding visualization.")
        return

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(5, X.shape[0] - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(X)
    elif method == "umap":
        try:
            import umap
            n_neighbors = min(5, X.shape[0] - 1)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            coords = reducer.fit_transform(X)
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE.")
            from sklearn.manifold import TSNE
            perplexity = min(5, X.shape[0] - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            coords = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Color mapping
    class_colors = {
        "wild_type": "#2ca02c",
        "contact": "#d62728",
        "conformational": "#1f77b4",
        "unknown": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(allele_names):
        cls = structural_classes.get(name, "unknown")
        color = class_colors.get(cls, "#7f7f7f")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=150, zorder=3, edgecolors="black")
        ax.annotate(
            name, (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold",
        )

    # Legend
    for cls, color in class_colors.items():
        if any(structural_classes.get(a) == cls for a in allele_names):
            ax.scatter([], [], c=color, s=100, label=cls.replace("_", " ").title(), edgecolors="black")
    ax.legend(loc="best", framealpha=0.9)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title("ESM-2 Allele Embedding Space")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"allele_embeddings_{method}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path / f"allele_embeddings_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved allele embedding plot ({method})")


def plot_grn_network(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    gene_names: Optional[List[str]] = None,
    edge_modes: Optional[Dict[int, str]] = None,
    highlight_edges: Optional[List[int]] = None,
    title: str = "GRN",
    output_dir: str = "figures",
    filename: str = "grn_network",
):
    """
    Visualize GRN as a directed network graph.

    Args:
        edge_index: (2, E) edge indices.
        edge_weights: (E,) edge weights.
        gene_names: Optional node labels.
        edge_modes: Dict mapping edge_idx -> "activating"/"repressive"/"mixed".
        highlight_edges: List of edge indices to highlight (allele-specific).
        title: Plot title.
        output_dir: Output directory.
        filename: Output filename (without extension).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    E = edge_index.shape[1]
    N = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1

    G = nx.DiGraph()
    for i in range(N):
        label = gene_names[i] if gene_names else str(i)
        G.add_node(i, label=label)

    mode_colors = {
        "activating": "#2ca02c",
        "repressive": "#d62728",
        "mixed": "#ff7f0e",
    }

    for e in range(E):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        w = edge_weights[e].item()
        mode = edge_modes.get(e, "mixed") if edge_modes else "mixed"
        color = mode_colors.get(mode, "#7f7f7f")

        if highlight_edges and e in highlight_edges:
            color = "#9467bd"  # Purple for allele-specific
            width = 3.0
        else:
            width = max(0.5, min(3.0, abs(w) * 2))

        G.add_edge(u, v, weight=w, color=color, width=width)

    # Layout
    if N < 50:
        pos = nx.spring_layout(G, k=2 / np.sqrt(N), iterations=100, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges
    edge_colors = [G[u][v]["color"] for u, v in G.edges()]
    edge_widths = [G[u][v]["width"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors, width=edge_widths,
        alpha=0.6, arrows=True, arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes
    node_sizes = []
    for i in range(N):
        degree = G.degree(i)
        node_sizes.append(100 + degree * 30)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes, node_color="#4C72B0",
        alpha=0.8, edgecolors="black", linewidths=0.5,
    )

    # Labels (only for small networks)
    if N <= 50:
        labels = {i: gene_names[i] if gene_names else str(i) for i in range(N)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    # Legend
    for mode, color in mode_colors.items():
        ax.plot([], [], color=color, linewidth=2, label=mode)
    if highlight_edges:
        ax.plot([], [], color="#9467bd", linewidth=3, label="allele-specific")
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved GRN network plot: {filename}")


def plot_therapeutic_window(
    bottlenecks: List[Dict],
    allele: str = "unknown",
    output_dir: str = "figures",
):
    """
    Visualize therapeutic window from transient bottleneck analysis.

    Shows centrality change magnitude over pseudotime, highlighting
    time points where in silico intervention would be most effective.
    """
    import matplotlib.pyplot as plt

    if not bottlenecks:
        logger.warning("No bottleneck data to plot.")
        return

    # Sort by time index
    sorted_bn = sorted(bottlenecks, key=lambda x: x["time_index"])
    times = [b["time_index"] for b in sorted_bn]
    changes = [b["centrality_change"] for b in sorted_bn]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bar chart of centrality changes
    colors = ["#d62728" if c > np.median(changes) else "#4C72B0" for c in changes]
    ax.bar(times, changes, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Highlight therapeutic window (top quartile)
    threshold = np.percentile(changes, 75)
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5,
               label=f"Therapeutic threshold (Q75={threshold:.3f})")

    ax.set_xlabel("Pseudotime bin transition")
    ax.set_ylabel("Betweenness centrality change")
    ax.set_title(f"Therapeutic window identification - {allele}")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"therapeutic_window_{allele}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved therapeutic window plot for {allele}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test allele embedding visualization
    esm_embeddings = {
        "WT": torch.randn(1280),
        "R175H": torch.randn(1280),
        "R273H": torch.randn(1280),
        "R248W": torch.randn(1280),
        "G245S": torch.randn(1280),
    }

    print("Visualization modules ready.")
    print(f"Alleles available: {list(esm_embeddings.keys())}")
    print("Call plot functions with matplotlib installed for figure generation.")
