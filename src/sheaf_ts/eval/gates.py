"""Fail-closed scientific gates for shortcut-resistant evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from sheaf_ts.eval.stats import bootstrap_mean_ci

__all__ = ["GateThresholds", "evaluate_scientific_gates"]


@dataclass(frozen=True)
class GateThresholds:
    """Minimum margins required over controls."""

    primary_margin: float = 0.05
    mechanism_margin: float = 0.02
    minimum_model_metric: float | None = 0.0
    paired_ci_confidence: float = 0.95
    minimum_margin_samples: int = 2


def _finite(name: str, value: float) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(observed):
        raise ValueError(f"{name} must be finite")
    return observed


def evaluate_scientific_gates(
    *,
    model_metric: float,
    controls: Mapping[str, float],
    required: Sequence[str],
    thresholds: GateThresholds | None = None,
    paired_margins: Mapping[str, Sequence[float]] | None = None,
) -> dict:
    """Evaluate model-vs-control margins.

    Required controls fail closed when missing, non-finite, or too close to the
    model. ``mean_drift`` uses ``primary_margin``; all other controls use
    ``mechanism_margin``.
    """
    thresholds = thresholds or GateThresholds()
    model = _finite("model_metric", model_metric)
    checks = {}
    failed = []

    if thresholds.minimum_model_metric is not None and model < thresholds.minimum_model_metric:
        checks["minimum_model_metric"] = {
            "status": "fail",
            "passed": False,
            "reason": "model metric below absolute floor",
            "control_metric": None,
            "margin": model - thresholds.minimum_model_metric,
            "threshold": thresholds.minimum_model_metric,
        }
        failed.append("minimum_model_metric")

    for name in required:
        if name not in controls:
            checks[name] = {
                "status": "fail",
                "passed": False,
                "reason": "missing control",
                "control_metric": None,
                "margin": None,
                "threshold": None,
            }
            failed.append(name)
            continue

        control = _finite(f"control {name}", controls[name])
        threshold = (
            thresholds.primary_margin
            if name == "mean_drift"
            else thresholds.mechanism_margin
        )
        margin = model - control
        ci = None
        if paired_margins is not None and name in paired_margins:
            samples = list(paired_margins[name])
            if len(samples) < thresholds.minimum_margin_samples:
                ci = {
                    "low": None,
                    "high": None,
                    "mean": None,
                    "n": float(len(samples)),
                    "confidence": thresholds.paired_ci_confidence,
                }
                passed = False
            else:
                ci = bootstrap_mean_ci(
                    samples,
                    confidence=thresholds.paired_ci_confidence,
                    seed=0,
                )
                passed = margin >= threshold and ci["low"] >= threshold
        else:
            passed = margin >= threshold
        checks[name] = {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "control_metric": control,
            "margin": margin,
            "threshold": threshold,
            "ci_low": None if ci is None else ci["low"],
            "ci_high": None if ci is None else ci["high"],
            "ci_n": None if ci is None else ci["n"],
            "reason": "model clears control margin" if passed else "control too close to model",
        }
        if not passed:
            failed.append(name)

    return {
        "accepted": not failed,
        "status": "ACCEPT" if not failed else "REJECT",
        "model_metric": model,
        "failed_controls": failed,
        "thresholds": asdict(thresholds),
        "checks": checks,
    }
