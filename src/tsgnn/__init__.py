"""
TS-GNN: Temporal Sheaf Graph Neural Network.

Allele-conditioned gene regulatory network rewiring
via temporal sheaf diffusion on pseudotime-ordered single-cell transcriptomes.

Public API
----------
- TSGNN: Main model class
- SheafDiffusionLayer: Core sheaf diffusion layer
- VectorizedSheafDiffusion: Optimized vectorized sheaf layer
- TemporalRestrictionEvolution: GRU for restriction map evolution
- AlleleFiLM: FiLM allele conditioning module
- AbstractGRNModel: Abstract base class for all GRN models
- ModelRegistry / MetricRegistry: Plugin-based registries
- TSGNNConfig: Typed configuration schema
- TSGNNLoss: Composite 4-term loss function
- TSGNNTrainer: Training pipeline
"""

__version__ = "0.2.0"

from tsgnn._compat import apply_platform_fixes

apply_platform_fixes()

from tsgnn.model.tsgnn import TSGNN, create_tsgnn_from_config
from tsgnn.model.sheaf import SheafDiffusionLayer
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
from tsgnn.model.temporal import TemporalRestrictionEvolution
from tsgnn.model.film import AlleleFiLM, ZeroFiLM
from tsgnn.model.base_model import AbstractGRNModel
from tsgnn.model.registry import ModelRegistry, MetricRegistry
from tsgnn.config import TSGNNConfig
from tsgnn.training.loss import TSGNNLoss
from tsgnn.training.trainer import TSGNNTrainer

__all__ = [
    "TSGNN",
    "create_tsgnn_from_config",
    "SheafDiffusionLayer",
    "VectorizedSheafDiffusion",
    "TemporalRestrictionEvolution",
    "AlleleFiLM",
    "ZeroFiLM",
    "AbstractGRNModel",
    "ModelRegistry",
    "MetricRegistry",
    "TSGNNConfig",
    "TSGNNLoss",
    "TSGNNTrainer",
]
