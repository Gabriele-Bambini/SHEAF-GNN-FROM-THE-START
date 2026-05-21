"""Evaluation utilities for the clean sheaf rebuild."""

from sheaf_ts.eval.artifacts import ControlResult, ExperimentResult
from sheaf_ts.eval.cross_validation import CVSplit, repeated_group_kfold, temporal_blocked_splits
from sheaf_ts.eval.gates import GateThresholds, evaluate_scientific_gates
from sheaf_ts.eval.metrics import (
    directional_metrics,
    fit_mean_drift,
    partial_pearsonr_flat,
    pearsonr_flat,
    residualize_against,
)
from sheaf_ts.eval.stats import benjamini_hochberg, bootstrap_mean_ci

__all__ = [
    "CVSplit",
    "ControlResult",
    "ExperimentResult",
    "GateThresholds",
    "directional_metrics",
    "benjamini_hochberg",
    "bootstrap_mean_ci",
    "evaluate_scientific_gates",
    "fit_mean_drift",
    "partial_pearsonr_flat",
    "pearsonr_flat",
    "residualize_against",
    "repeated_group_kfold",
    "temporal_blocked_splits",
]
