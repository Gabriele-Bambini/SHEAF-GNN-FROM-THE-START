"""
Typed configuration schema for TS-GNN.

REPLACES the raw dict-based config pattern with @dataclass + validation.
This makes configuration errors caught at initialization rather than
buried in training failures.

Impact:
  - Code Quality: dict → typed dataclass (IDE autocomplete, type checking)
  - Usability: informative error messages on invalid config combinations
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    K: int = 15                         # Number of pseudotime bins
    binning_method: str = "adaptive"    # "quantile" or "adaptive"
    n_genes: int = 500                  # Number of genes (N)
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_pct_mito: float = 20.0
    batch_key: Optional[str] = None     # Batch correction key
    n_pcs: int = 50                     # PCA components
    data_dir: str = "data"

    def validate(self):
        if self.K < 2:
            raise ValueError(f"data.K must be ≥ 2 (got {self.K}); need at least 2 pseudotime bins.")
        if self.n_genes < 10:
            raise ValueError(f"data.n_genes must be ≥ 10 (got {self.n_genes}).")
        if not (0 < self.max_pct_mito < 100):
            raise ValueError(f"data.max_pct_mito must be in (0, 100) (got {self.max_pct_mito}).")
        if self.min_genes_per_cell < 1:
            raise ValueError(f"data.min_genes_per_cell must be ≥ 1 (got {self.min_genes_per_cell}).")
        if self.min_cells_per_gene < 1:
            raise ValueError(f"data.min_cells_per_gene must be ≥ 1 (got {self.min_cells_per_gene}).")
        if self.n_pcs < 1:
            raise ValueError(f"data.n_pcs must be ≥ 1 (got {self.n_pcs}).")


@dataclass
class ModelConfig:
    """Configuration for TS-GNN model architecture."""
    stalk_dim: int = 6                  # d: sheaf stalk dimension
    input_dim: int = 500                # Gene feature dimension (must match data.n_genes)
    esm_dim: int = 1280                 # ESM-2 embedding dimension
    conditioning_dim: int = 128         # FiLM conditioning latent dim
    num_diffusion_steps: int = 3        # Sheaf diffusion iterations per t
    use_allele_conditioning: bool = True
    num_edges: int = 200                # E: number of regulatory edges

    def validate(self):
        if self.stalk_dim < 2:
            raise ValueError(
                f"model.stalk_dim must be ≥ 2 (got {self.stalk_dim}). "
                f"d=1 reduces sheaf to standard graph Laplacian (no geometric benefit)."
            )
        if self.num_diffusion_steps < 1:
            raise ValueError(
                f"model.num_diffusion_steps must be ≥ 1 (got {self.num_diffusion_steps})."
            )
        if self.esm_dim <= 0:
            raise ValueError(f"model.esm_dim must be positive (got {self.esm_dim}).")
        if self.conditioning_dim <= 0:
            raise ValueError(f"model.conditioning_dim must be positive (got {self.conditioning_dim}).")
        if self.num_edges < 1:
            raise ValueError(f"model.num_edges must be ≥ 1 (got {self.num_edges}).")
        if self.stalk_dim > 8:
            import warnings
            warnings.warn(
                f"stalk_dim={self.stalk_dim} is unusually large. "
                f"Laplacian will be {self.input_dim * self.stalk_dim}×"
                f"{self.input_dim * self.stalk_dim} = "
                f"{(self.input_dim * self.stalk_dim)**2 / 1e6:.1f}M entries. "
                f"Consider d ∈ [2, 8] for efficiency.",
                stacklevel=2,
            )


@dataclass
class LossConfig:
    """Configuration for the composite loss function."""
    lambda_1: float = 0.1              # Topological preservation weight
    lambda_2: float = 0.0              # Regulon activity weight (enable when DoRothEA data present)
    lambda_3: float = 0.01             # Sparsity weight
    lambda_4: float = 0.0              # RNA velocity consistency weight
    tau: float = 0.5                   # Hinge threshold for topological loss
    neuralsort_temperature: float = 1.0 # Temperature for soft-ranking AFTER magnitude normalization

    def validate(self):
        for name, val in [
            ("lambda_1", self.lambda_1),
            ("lambda_2", self.lambda_2),
            ("lambda_3", self.lambda_3),
            ("lambda_4", self.lambda_4),
        ]:
            if val < 0:
                raise ValueError(f"loss.{name} must be ≥ 0 (got {val}).")
        if self.tau <= 0:
            raise ValueError(f"loss.tau must be > 0 (got {self.tau}).")
        if self.lambda_2 > 0 and self.lambda_3 > 0:
            import warnings
            warnings.warn(
                "Both regulon (λ₂) and sparsity (λ₃) losses are active. "
                "Ensure ULM regulon activity matrix and allele embeddings with grad are provided.",
                stacklevel=2,
            )


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    lr: float = 1e-3
    max_epochs: int = 500
    patience: int = 20                  # Early stopping patience
    max_grad_norm: float = 1.0          # Gradient clipping
    weight_decay: float = 0.0
    mixed_precision: bool = True
    use_wandb: bool = False
    wandb_project: str = "tsgnn"
    run_name: Optional[str] = None
    resume_from_checkpoint: bool = False  # Opt-in checkpoint resume (safer default)
    checkpoint_dir: str = "checkpoints"
    deterministic: bool = False           # Bit-for-bit reproducibility (slower)

    def validate(self):
        if self.lr <= 0:
            raise ValueError(f"training.lr must be > 0 (got {self.lr}).")
        if self.max_epochs < 1:
            raise ValueError(f"training.max_epochs must be ≥ 1 (got {self.max_epochs}).")
        if self.patience < 1:
            raise ValueError(f"training.patience must be ≥ 1 (got {self.patience}).")
        if self.max_grad_norm <= 0:
            raise ValueError(f"training.max_grad_norm must be > 0 (got {self.max_grad_norm}).")
        if self.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be ≥ 0 (got {self.weight_decay}).")
        if self.lr > 0.1:
            import warnings
            warnings.warn(
                f"lr={self.lr} is very high for sheaf diffusion. "
                f"Consider lr ∈ [1e-4, 1e-2] to avoid gradient explosion.",
                stacklevel=2,
            )


@dataclass
class AblationConfig:
    """Configuration for ablation study parameters."""
    stalk_dims: List[int] = field(default_factory=lambda: [4, 6, 8])
    diffusion_steps: List[int] = field(default_factory=lambda: [1, 3, 5])
    conditioning_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class TSGNNConfig:
    """Top-level configuration combining all sub-configs."""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def validate(self):
        """Validate all sub-configs and cross-config constraints."""
        self.data.validate()
        self.model.validate()
        self.loss.validate()
        self.training.validate()

        # Cross-config validation
        if self.model.input_dim != self.data.n_genes:
            import warnings
            warnings.warn(
                f"model.input_dim ({self.model.input_dim}) ≠ data.n_genes "
                f"({self.data.n_genes}). These should typically match.",
                stacklevel=2,
            )

    @classmethod
    def from_dict(cls, d: dict) -> "TSGNNConfig":
        """Create config from a nested dictionary (e.g., loaded from YAML).

        Unknown keys in each sub-section are logged as warnings rather than
        silently dropped — this catches typos like ``lr_max`` instead of
        ``lr`` before they turn into subtle training bugs.
        """
        import logging
        log = logging.getLogger(__name__)

        def _filter(section_name: str, payload: dict, schema_cls) -> dict:
            fields = schema_cls.__dataclass_fields__
            kept: dict = {}
            for k, v in payload.items():
                if k in fields:
                    kept[k] = v
                else:
                    log.warning(
                        "Unknown config key '%s.%s' ignored. Valid keys: %s",
                        section_name, k, sorted(fields.keys()),
                    )
            return kept

        config = cls(
            seed=d.get("seed", 42),
            data=DataConfig(**_filter("data", d.get("data", {}) or {}, DataConfig)),
            model=ModelConfig(**_filter("model", d.get("model", {}) or {}, ModelConfig)),
            loss=LossConfig(**_filter("loss", d.get("loss", {}) or {}, LossConfig)),
            training=TrainingConfig(**_filter("training", d.get("training", {}) or {}, TrainingConfig)),
            ablation=AblationConfig(**_filter("ablation", d.get("ablation", {}) or {}, AblationConfig)),
        )
        config.validate()
        return config

    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint saving / wandb logging."""
        from dataclasses import asdict
        return asdict(self)
