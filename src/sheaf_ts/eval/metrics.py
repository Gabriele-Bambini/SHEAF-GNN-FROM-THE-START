"""Robust directional metrics for time-series gene prediction."""

from __future__ import annotations

import torch

__all__ = [
    "directional_metrics",
    "fit_mean_drift",
    "partial_pearsonr_flat",
    "pearsonr_flat",
    "residualize_against",
]


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.detach().to(dtype=torch.float64)


def pearsonr_flat(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Finite Pearson correlation over flattened tensors.

    Returns ``0.0`` for degenerate vectors instead of producing NaN.
    """
    x = _as_float_tensor(a).reshape(-1)
    y = _as_float_tensor(b).reshape(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    if valid.sum().item() < 2:
        return 0.0
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom <= eps:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def fit_mean_drift(train_delta_true: torch.Tensor) -> torch.Tensor:
    """Fit the mean drift vector using training transitions only."""
    train = _as_float_tensor(train_delta_true)
    if train.ndim == 0:
        raise ValueError("train_delta_true must have at least one dimension")
    if not torch.isfinite(train).all():
        raise ValueError("train_delta_true contains non-finite values")
    return train.reshape(-1, train.shape[-1]).mean(dim=0)


def _project_out(vector: torch.Tensor, direction: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    norm_sq = direction @ direction
    if norm_sq <= eps:
        return vector
    return vector - ((vector @ direction) / norm_sq) * direction


def _project_rows_out(matrix: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    rows = matrix.reshape(-1, matrix.shape[-1])
    projected = torch.stack([_project_out(row, direction) for row in rows], dim=0)
    return projected.reshape_as(matrix)


def residualize_against(values: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
    """Remove the best linear prediction from ``covariates`` row-wise.

    This catches identity/autocorrelation shortcuts: if a predictor only learns
    "delta is a function of current x", its residual correlation should vanish.
    """
    y = _as_float_tensor(values)
    x = _as_float_tensor(covariates)
    if y.shape[:-1] != x.shape[:-1]:
        raise ValueError("values and covariates must share all leading dimensions")
    y_rows = y.reshape(-1, y.shape[-1])
    x_rows = x.reshape(-1, x.shape[-1])
    design = torch.cat(
        [torch.ones(x_rows.shape[0], 1, dtype=x_rows.dtype, device=x_rows.device), x_rows],
        dim=1,
    )
    coeff = torch.linalg.pinv(design) @ y_rows
    residual = y_rows - design @ coeff
    return residual.reshape_as(y)


def partial_pearsonr_flat(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    covariates: torch.Tensor,
) -> float:
    """Pearson correlation after linear residualization against covariates."""
    return pearsonr_flat(
        residualize_against(a, covariates),
        residualize_against(b, covariates),
    )


def directional_metrics(
    delta_pred: torch.Tensor,
    delta_true: torch.Tensor,
    *,
    mean_drift: torch.Tensor,
    current_state: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute raw and mean-drift-orthogonal directional correlations."""
    pred = _as_float_tensor(delta_pred)
    true = _as_float_tensor(delta_true)
    drift = _as_float_tensor(mean_drift).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"delta_pred shape {tuple(pred.shape)} != delta_true {tuple(true.shape)}")
    if pred.shape[-1] != drift.numel():
        raise ValueError("mean_drift dimension must match last tensor dimension")
    if not torch.isfinite(pred).all() or not torch.isfinite(true).all():
        raise ValueError("delta tensors contain non-finite values")

    pred_perp = _project_rows_out(pred, drift)
    true_perp = _project_rows_out(true, drift)
    metrics = {
        "r_delta": pearsonr_flat(pred, true),
        "r_delta_perp": pearsonr_flat(pred_perp, true_perp),
        "mean_drift_norm": float(torch.linalg.norm(drift).item()),
    }
    if current_state is not None:
        state = _as_float_tensor(current_state)
        metrics["r_delta_no_self"] = partial_pearsonr_flat(pred, true, covariates=state)
        metrics["r_delta_perp_no_self"] = partial_pearsonr_flat(
            pred_perp,
            true_perp,
            covariates=state,
        )
    return metrics
