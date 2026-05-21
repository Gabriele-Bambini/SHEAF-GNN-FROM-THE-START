"""
Dependency-injected Model and Metric Registries.

REPLACES hardcoded model instantiation and metric function calls with
a plugin-based registry pattern. This enables:

1. Decoupling trainer from model implementation
2. Easy addition of new baselines without modifying existing code
3. Metric extensibility for custom evaluation criteria

Impact:
  - Architecture: dependency injection, Open-Closed Principle
  - Usability: `register("my_model", MyModel)` instead of editing REGISTRY dict
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Model Registry ──────────────────────────────────────────────────────

class ModelRegistry:
    """
    Registry for GRN model factories.

    Usage:
        registry = ModelRegistry()
        registry.register("tsgnn", TSGNN)
        registry.register("evolvegcn", EvolveGCN)

        model = registry.create("tsgnn", num_nodes=50, ...)
    """

    def __init__(self):
        self._factories: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str, factory: Type[nn.Module]) -> None:
        """Register a model class or factory function."""
        if name in self._factories:
            logger.warning(f"Overwriting model '{name}' in registry.")
        self._factories[name] = factory
        logger.debug(f"Registered model: {name}")

    def create(self, name: str, **kwargs) -> nn.Module:
        """Create a model instance by name."""
        if name not in self._factories:
            available = ", ".join(sorted(self._factories.keys()))
            raise ValueError(
                f"Unknown model '{name}'. Available models: [{available}]. "
                f"Register new models with registry.register(name, ModelClass)."
            )
        return self._factories[name](**kwargs)

    def list(self) -> List[str]:
        """List all registered model names."""
        return sorted(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)


# ── Metric Registry ─────────────────────────────────────────────────────

class MetricRegistry:
    """
    Registry for evaluation metric functions.

    Usage:
        metrics = MetricRegistry()
        metrics.register("zero_shot", cross_cancer_zero_shot)
        metrics.register("enrichment", allele_edge_enrichment)

        results = metrics.run_all(shared_kwargs)
    """

    def __init__(self):
        self._metrics: Dict[str, Callable] = {}
        self._metric_kwargs: Dict[str, Dict] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        default_kwargs: Optional[Dict] = None,
    ) -> None:
        """Register an evaluation metric function."""
        self._metrics[name] = fn
        self._metric_kwargs[name] = default_kwargs or {}
        logger.debug(f"Registered metric: {name}")

    def run(self, name: str, **kwargs) -> Dict[str, Any]:
        """Run a single metric by name."""
        if name not in self._metrics:
            raise ValueError(f"Unknown metric '{name}'.")
        merged = {**self._metric_kwargs[name], **kwargs}
        return self._metrics[name](**merged)

    def run_all(self, **shared_kwargs) -> Dict[str, Dict]:
        """Run all registered metrics with shared keyword arguments."""
        results = {}
        for name, fn in self._metrics.items():
            try:
                merged = {**self._metric_kwargs[name], **shared_kwargs}
                # Only pass kwargs that the function accepts
                import inspect
                sig = inspect.signature(fn)
                valid_kwargs = {k: v for k, v in merged.items() if k in sig.parameters}
                results[name] = fn(**valid_kwargs)
            except Exception as e:
                logger.warning(f"Metric '{name}' failed: {e}")
                results[name] = {"error": str(e)}
        return results

    def list(self) -> List[str]:
        return sorted(self._metrics.keys())


# ── Factory Function ────────────────────────────────────────────────────

def create_default_model_registry() -> ModelRegistry:
    """Create a ModelRegistry with all built-in models pre-registered."""
    registry = ModelRegistry()

    # Lazy imports to avoid circular dependencies
    try:
        from tsgnn.model.tsgnn import TSGNN
        registry.register("tsgnn", TSGNN)
    except ImportError:
        pass

    try:
        from tsgnn.model.baselines import (
            EvolveGCN, TemporalGAT, create_tsgnn_no_allele
        )
        registry.register("evolvegcn", EvolveGCN)
        registry.register("temporal_gat", TemporalGAT)
        registry.register("tsgnn_no_allele", create_tsgnn_no_allele)
    except ImportError:
        pass

    return registry


def create_default_metric_registry() -> MetricRegistry:
    """Create a MetricRegistry with all built-in metrics pre-registered."""
    registry = MetricRegistry()

    try:
        from tsgnn.evaluation.metrics import (
            cross_cancer_zero_shot,
            allele_edge_enrichment,
            chromatin_remodeler_concordance,
            depmap_dependency_correlation,
            rewiring_distinguishability,
        )
        registry.register("zero_shot", cross_cancer_zero_shot)
        registry.register("enrichment", allele_edge_enrichment)
        registry.register("chip_seq", chromatin_remodeler_concordance)
        registry.register("depmap", depmap_dependency_correlation)
        registry.register("distinguishability", rewiring_distinguishability)
    except ImportError:
        pass

    return registry
