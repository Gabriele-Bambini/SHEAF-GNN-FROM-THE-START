"""
AlleleSO4Embedding -- learnable SO(d) rotation per allele class.

Each TP53 allele class is a rotation R_c in SO(d) applied to all restriction maps:
    F_allele[e] = F_base[e] @ R_c

R_c is parameterized via its Lie algebra:
    params_c in R^{d(d-1)/2}  ->  A_c = skew(params_c)  ->  R_c = exp(A_c) in SO(d)

This is the minimal allele-conditioning mechanism:
- 18 total parameters (3 classes x 6 dims of so(4))
- R_c stays on SO(d) manifold by construction
- Gradients flow cleanly through Cayley map
- Interpretable: SO(4) pairwise distances between classes = biological similarity

Usage:
    emb = AlleleSO4Embedding(n_allele_classes=3, stalk_dim=4)
    R = emb(allele_class_tensor)            # (d, d) rotation matrix
    F_allele = torch.einsum('esij,jk->esik', F_base, R)  # (E, 2, d, d)
"""

import torch
import torch.nn as nn

# Reuse existing implementations -- do NOT duplicate
from .sheaf_perturbation import skew_from_params, exp_skew

__all__ = ["AlleleSO4Embedding"]


class AlleleSO4Embedding(nn.Module):
    """
    Learnable SO(d) rotation per allele class.

    Args:
        n_allele_classes: Number of distinct allele classes (default 3).
        stalk_dim: d -- sheaf stalk dimension (default 4).
        init_scale: std for random init. Small (0.1) -> starts near identity.
    """

    def __init__(
        self,
        n_allele_classes: int = 3,
        stalk_dim: int = 4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_allele_classes
        self.d = stalk_dim
        self.n_skew = stalk_dim * (stalk_dim - 1) // 2

        # One skew-parameter vector per allele class
        self.skew_params = nn.Embedding(n_allele_classes, self.n_skew)
        nn.init.normal_(self.skew_params.weight, std=init_scale)

        # WT (class 0) starts exactly at identity (zero skew = identity rotation)
        with torch.no_grad():
            self.skew_params.weight[0].zero_()

    def forward(self, allele_class: torch.Tensor) -> torch.Tensor:
        """
        Return SO(d) rotation matrix for the given allele class.

        Args:
            allele_class: scalar LongTensor in [0, n_classes).

        Returns:
            R: (d, d) orthogonal matrix with det=+1.
        """
        params = self.skew_params(allele_class)   # (n_skew,)
        A = skew_from_params(params.unsqueeze(0), self.d).squeeze(0)  # (d, d)
        R = exp_skew(A.unsqueeze(0)).squeeze(0)   # (d, d)
        return R

    def all_rotations(self) -> torch.Tensor:
        """Return (n_classes, d, d) rotation matrices for all classes."""
        device = self.skew_params.weight.device
        ids = torch.arange(self.n_classes, device=device)
        Rs = []
        for i in ids:
            Rs.append(self.forward(i))
        return torch.stack(Rs)  # (n_classes, d, d)

    def pairwise_so_distances(self) -> torch.Tensor:
        """
        Compute SO(d) pairwise distances between all allele classes.

        Distance: ||R_i - R_j||_F / sqrt(2*d) in [0, 1]
        (0 = identical rotations, 1 = maximally different)

        Returns:
            dist: (n_classes, n_classes) symmetric distance matrix.
        """
        Rs = self.all_rotations()  # (n_classes, d, d)
        n = self.n_classes
        dist = torch.zeros(n, n, device=Rs.device)
        # Max ||R_i - R_j||_F for R in SO(d) is 2*sqrt(d)
        norm_factor = 2.0 * (self.d ** 0.5)
        for i in range(n):
            for j in range(n):
                dist[i, j] = (Rs[i] - Rs[j]).norm() / norm_factor
        return dist
