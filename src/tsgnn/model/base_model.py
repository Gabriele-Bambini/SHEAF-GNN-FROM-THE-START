"""
Abstract base class for all GRN models in TS-GNN.

Enforces a uniform interface across TS-GNN and all baselines, making
interface uniformity a compile-time guarantee (via ABC) rather than a
runtime test assertion.

Impact:
  - Architecture: enforced contract via ABC
  - Code Quality: catch missing methods at class definition, not at call
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AbstractGRNModel(ABC, nn.Module):
    """
    Abstract base class for Gene Regulatory Network models.

    All models — TS-GNN, EvolveGCN, TemporalGAT, and future baselines —
    must implement this interface. This guarantees:

    1. Uniform forward() signature: (node_seq, allele_emb) → (preds, maps, laps)
    2. count_parameters() for reproducibility reporting
    3. predict() for inference-time convenience

    Subclasses that fail to implement abstract methods will raise
    TypeError at instantiation, not at runtime.
    """

    @abstractmethod
    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
        edge_weights_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            node_features_seq: (K, N, input_dim) temporal node features.
            allele_embedding: (esm_dim,) or (1, esm_dim) allele embedding.
            edge_weights_seq: Optional (K, E) temporal edge weights.

        Returns:
            predictions: List of K tensors, each (N, input_dim).
            maps_trajectory: List of K tensors (model-specific, e.g., restriction maps).
            laplacians: List of K tensors (model-specific, e.g., sheaf Laplacians).
        """
        ...

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters by module.

        Returns:
            Dict mapping module_name → parameter_count, with 'total' key.
        """
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                counts[name] = n
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    @torch.no_grad()
    def predict(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Inference-time prediction (no grad, returns predictions only).

        Args:
            node_features_seq: (K, N, input_dim) temporal node features.
            allele_embedding: (esm_dim,) allele embedding.

        Returns:
            predictions: List of K tensors, each (N, input_dim).
        """
        self.eval()
        preds, _, _ = self.forward(node_features_seq, allele_embedding)
        return preds

    def summary(self) -> str:
        """Human-readable model summary."""
        counts = self.count_parameters()
        lines = [f"Model: {self.__class__.__name__}"]
        for name, count in sorted(counts.items()):
            if name != "total":
                lines.append(f"  {name}: {count:,} params")
        lines.append(f"  TOTAL: {counts['total']:,} params")
        return "\n".join(lines)
