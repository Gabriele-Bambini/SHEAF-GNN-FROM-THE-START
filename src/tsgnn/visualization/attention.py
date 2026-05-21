"""
Restriction Map Analysis and Visualization.

Phase 5.2: Visualize learned restriction maps:
- Eigenvalue distribution (negative eigenvalues = repression)
- Temporal evolution of maps for specific TF-target pairs
- Heatmaps of restriction map matrices
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_restriction_map_eigenvalues(
    maps_trajectory: List[torch.Tensor],
    edge_index: torch.Tensor,
    gene_names: Optional[List[str]] = None,
    allele: str = "unknown",
    output_dir: str = "figures",
):
    """
    Visualize eigenvalue distribution of restriction maps over time.

    Negative eigenvalues indicate repressive regulatory relationships.
    Positive eigenvalues indicate activating relationships.

    Args:
        maps_trajectory: List of K tensors, each (E, 2, d, d).
        edge_index: (2, E) edge indices.
        gene_names: Optional gene name labels.
        allele: Allele name for title.
        output_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    K = len(maps_trajectory)
    E = maps_trajectory[0].shape[0]
    d = maps_trajectory[0].shape[2]

    # Collect eigenvalues across time and edges
    all_eigenvalues = []
    for t in range(K):
        maps = maps_trajectory[t].detach().cpu()
        eigs_t = []
        for e in range(E):
            for s in range(2):  # source and target maps
                M = maps[e, s]
                eigvals = torch.linalg.eigvalsh(M @ M.T)  # real eigenvalues
                eigs_t.extend(eigvals.numpy().tolist())
        all_eigenvalues.append(eigs_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Eigenvalue distribution over time (violin plot)
    ax = axes[0]
    positions = list(range(K))
    parts = ax.violinplot(all_eigenvalues, positions=positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("Eigenvalue (F^T F)")
    ax.set_title(f"Restriction map eigenvalue evolution - {allele}")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="zero line")
    ax.legend()

    # Panel B: Fraction of "repressive" eigenvalues over time
    ax = axes[1]
    neg_fractions = []
    for t in range(K):
        maps = maps_trajectory[t].detach().cpu()
        neg_count = 0
        total_count = 0
        for e in range(E):
            # Use the off-diagonal product F_src^T @ F_tgt
            F_src = maps[e, 0]
            F_tgt = maps[e, 1]
            product = F_src.T @ F_tgt
            eigvals = torch.linalg.eigvalsh(product)
            neg_count += (eigvals < 0).sum().item()
            total_count += d
        neg_fractions.append(neg_count / max(total_count, 1))

    ax.plot(range(K), neg_fractions, "o-", color="#C44E52", linewidth=2, markersize=8)
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("Fraction of negative eigenvalues")
    ax.set_title(f"Repressive edge proportion - {allele}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"restriction_eigenvalues_{allele}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved restriction map eigenvalue plot for {allele}")


def plot_restriction_map_heatmap(
    maps_trajectory: List[torch.Tensor],
    edge_index: torch.Tensor,
    edge_idx: int,
    gene_names: Optional[List[str]] = None,
    allele: str = "unknown",
    output_dir: str = "figures",
):
    """
    Plot heatmaps of restriction maps for a specific edge over time.

    Shows how F_{u,e} and F_{v,e} evolve, revealing changing regulatory mode.
    """
    import matplotlib.pyplot as plt

    K = len(maps_trajectory)
    d = maps_trajectory[0].shape[2]

    fig, axes = plt.subplots(2, K, figsize=(3 * K, 6))
    if K == 1:
        axes = axes.reshape(2, 1)

    u = edge_index[0, edge_idx].item()
    v = edge_index[1, edge_idx].item()
    edge_label = f"{gene_names[u]}->{gene_names[v]}" if gene_names else f"{u}->{v}"

    vmin, vmax = -2, 2
    for t in range(K):
        maps = maps_trajectory[t].detach().cpu()
        F_src = maps[edge_idx, 0].numpy()
        F_tgt = maps[edge_idx, 1].numpy()

        axes[0, t].imshow(F_src, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, t].set_title(f"t={t}", fontsize=9)
        if t == 0:
            axes[0, t].set_ylabel(f"F_src ({edge_label})")

        axes[1, t].imshow(F_tgt, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if t == 0:
            axes[1, t].set_ylabel(f"F_tgt ({edge_label})")

        for ax_row in [0, 1]:
            axes[ax_row, t].set_xticks([])
            axes[ax_row, t].set_yticks([])

    fig.suptitle(f"Restriction maps over time - {allele}, edge {edge_label}", y=1.02)
    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path / f"restriction_heatmap_{allele}_edge{edge_idx}.pdf",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"Saved restriction map heatmap for {allele}, edge {edge_idx}")


def analyze_regulatory_mode(
    maps_trajectory: List[torch.Tensor],
    edge_index: torch.Tensor,
    gene_names: Optional[List[str]] = None,
) -> Dict[int, Dict]:
    """
    Classify each edge as activating, repressing, or mixed based on
    restriction map eigenvalue signs.

    Returns:
        Dict mapping edge_idx -> {mode, confidence, eigenvalues_over_time}.
    """
    K = len(maps_trajectory)
    E = maps_trajectory[0].shape[0]
    d = maps_trajectory[0].shape[2]

    edge_analysis = {}
    for e in range(E):
        neg_fracs = []
        eig_traces = []
        for t in range(K):
            maps = maps_trajectory[t].detach().cpu()
            F_src = maps[e, 0]
            F_tgt = maps[e, 1]
            product = F_src.T @ F_tgt
            eigvals = torch.linalg.eigvalsh(product)
            neg_frac = (eigvals < 0).float().mean().item()
            neg_fracs.append(neg_frac)
            eig_traces.append(eigvals.numpy().tolist())

        avg_neg = np.mean(neg_fracs)
        if avg_neg > 0.7:
            mode = "repressive"
        elif avg_neg < 0.3:
            mode = "activating"
        else:
            mode = "mixed"

        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        label = f"{gene_names[u]}->{gene_names[v]}" if gene_names else f"{u}->{v}"

        edge_analysis[e] = {
            "mode": mode,
            "avg_neg_fraction": avg_neg,
            "confidence": abs(avg_neg - 0.5) * 2,  # 0 to 1
            "label": label,
            "eigenvalues_over_time": eig_traces,
        }

    return edge_analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    K, E, d = 10, 50, 4
    N = 20
    edge_index = torch.randint(0, N, (2, E))
    maps_traj = [torch.randn(E, 2, d, d) for _ in range(K)]

    analysis = analyze_regulatory_mode(maps_traj, edge_index)
    modes = [a["mode"] for a in analysis.values()]
    print(f"Edge modes: activating={modes.count('activating')}, "
          f"repressive={modes.count('repressive')}, mixed={modes.count('mixed')}")
    print("Visualization module ready.")
