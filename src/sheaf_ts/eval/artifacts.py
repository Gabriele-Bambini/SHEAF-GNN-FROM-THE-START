"""JSON-serializable experiment artifacts with fail-closed validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

__all__ = ["ControlResult", "ExperimentResult"]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_metric_mapping(metrics: Mapping[str, Any], *, prefix: str) -> None:
    if not metrics:
        raise ValueError(f"{prefix} metrics are empty")
    for name, value in metrics.items():
        if not _is_finite_number(value):
            raise ValueError(f"{prefix} metric {name!r} is missing or non-finite")


@dataclass(frozen=True)
class ControlResult:
    """Result of one destructive or baseline control."""

    name: str
    metric: float
    delta: float | None = None
    passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("control name is required")
        if not _is_finite_number(self.metric):
            raise ValueError(f"control {self.name!r} metric is non-finite")
        if self.delta is not None and not _is_finite_number(self.delta):
            raise ValueError(f"control {self.name!r} delta is non-finite")

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """Single fold/seed result for a model level."""

    run_id: str
    level: str
    seed: int
    fold: int
    metrics: dict[str, float]
    controls: dict[str, ControlResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, required_controls: tuple[str, ...] = ()) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.level:
            raise ValueError("level is required")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.fold, int):
            raise ValueError("fold must be an integer")
        _validate_metric_mapping(self.metrics, prefix="experiment")

        missing = [name for name in required_controls if name not in self.controls]
        if missing:
            raise ValueError("missing required controls: " + ", ".join(missing))
        for control in self.controls.values():
            control.validate()

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "level": self.level,
            "seed": self.seed,
            "fold": self.fold,
            "metrics": dict(self.metrics),
            "controls": {
                name: control.to_json_dict()
                for name, control in sorted(self.controls.items())
            },
            "metadata": dict(self.metadata),
        }
