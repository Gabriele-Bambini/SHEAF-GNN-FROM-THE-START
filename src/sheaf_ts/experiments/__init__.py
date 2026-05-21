"""Controlled experiments for validating the clean sheaf stack."""

from sheaf_ts.experiments.synthetic_sheaf import (
    SyntheticBacktestResult,
    SyntheticSheafDataset,
    backtest_oracle_sheaf,
    generate_sheaf_diffusion_dataset,
    synthetic_manifest,
    tune_tau_with_optuna,
)
from sheaf_ts.experiments.synthetic_runner import run_synthetic_backtest

__all__ = [
    "SyntheticBacktestResult",
    "SyntheticSheafDataset",
    "backtest_oracle_sheaf",
    "generate_sheaf_diffusion_dataset",
    "run_synthetic_backtest",
    "synthetic_manifest",
    "tune_tau_with_optuna",
]
