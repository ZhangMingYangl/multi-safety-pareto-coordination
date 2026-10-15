"""Fixed-bound metric normalization for risk and quality state values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableSequence


@dataclass(frozen=True)
class MetricSpec:
    lower: float
    upper: float
    invert: bool = False
    neutral: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.neutral <= 1:
            raise ValueError("neutral must be in [0, 1]")


class StateAdapter:
    """Normalize named metrics with bounds frozen before the final experiment."""

    def __init__(self, specs: Mapping[str, MetricSpec], eps: float = 1e-12) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.specs = dict(specs)
        self.eps = eps
        self.anomalies: MutableSequence[str] = []

    def normalize_one(self, name: str, raw_value: float) -> float:
        spec = self.specs[name]
        span = spec.upper - spec.lower
        if span < self.eps:
            self.anomalies.append(f"{name}: degenerate normalization bounds")
            return spec.neutral
        normalized = min(1.0, max(0.0, (float(raw_value) - spec.lower) / span))
        return 1.0 - normalized if spec.invert else normalized

    def adapt(self, raw_metrics: Mapping[str, float]) -> Dict[str, float]:
        missing = self.specs.keys() - raw_metrics.keys()
        if missing:
            raise KeyError(f"missing metrics: {sorted(missing)}")
        return {name: self.normalize_one(name, raw_metrics[name]) for name in self.specs}

