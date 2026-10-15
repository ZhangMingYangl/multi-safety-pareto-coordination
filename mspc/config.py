"""Configuration objects for MSPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MSPCConfig:
    """Numerical and algorithmic settings shared by core and full modes."""

    mode: str = "core"
    eps_grad: float = 1e-8
    eps_obj: float = 1e-12
    lambda_r: float = 1.0 / 3.0
    lambda_d: float = 1.0 / 3.0
    lambda_u: float = 1.0 / 3.0
    history_size: int = 5
    grid_step: float = 0.05
    projection_order: str = "conflict_desc"
    gradient_scaling: str = "unit"
    descent_guard: str = "none"
    descent_tolerance: float = 1e-7
    min_task_weight: float = 0.0
    alpha_smoothing: float = 0.0
    fused_scale: str = "unit"
    weight_update_interval: int = 1
    risk_thresholds: Optional[Sequence[float]] = None
    sampled_candidates: int = 2048
    sampling_seed: int = 0

    # Full-mode feedback settings.
    utility_min: float = 0.0
    stability_min: float = 0.0
    feedback_beta: float = 0.5
    lr_decay: float = 0.5
    min_lr: float = 1e-8
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"core", "full"}:
            raise ValueError("mode must be 'core' or 'full'")
        if self.eps_grad <= 0 or self.eps_obj <= 0:
            raise ValueError("numerical epsilons must be positive")
        lambdas = (self.lambda_r, self.lambda_d, self.lambda_u)
        if any(value < 0 for value in lambdas):
            raise ValueError("balance coefficients must be non-negative")
        if abs(sum(lambdas) - 1.0) > 1e-8:
            raise ValueError("lambda_r + lambda_d + lambda_u must equal 1")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if not 0 < self.grid_step <= 1:
            raise ValueError("grid_step must be in (0, 1]")
        reciprocal = 1.0 / self.grid_step
        if abs(reciprocal - round(reciprocal)) > 1e-8:
            raise ValueError("grid_step must divide 1 exactly")
        if self.projection_order != "conflict_desc":
            raise ValueError("only deterministic conflict_desc projection is supported")
        if self.gradient_scaling not in {"unit", "preserve"}:
            raise ValueError("gradient_scaling must be 'unit' or 'preserve'")
        if self.descent_guard not in {"none", "mgda"}:
            raise ValueError("descent_guard must be 'none' or 'mgda'")
        if self.descent_tolerance < 0:
            raise ValueError("descent_tolerance must be non-negative")
        if not 0 <= self.min_task_weight < 0.5:
            raise ValueError("min_task_weight must be in [0, 0.5)")
        if not 0 <= self.alpha_smoothing < 1:
            raise ValueError("alpha_smoothing must be in [0, 1)")
        if self.fused_scale not in {"unit", "task_count"}:
            raise ValueError("fused_scale must be 'unit' or 'task_count'")
        if self.weight_update_interval <= 0:
            raise ValueError("weight_update_interval must be positive")
        if self.sampled_candidates <= 0:
            raise ValueError("sampled_candidates must be positive")
        if self.risk_thresholds is not None:
            if any(not 0 <= value < 1 for value in self.risk_thresholds):
                raise ValueError("risk thresholds must be in [0, 1)")
        if self.feedback_beta < 0:
            raise ValueError("feedback_beta must be non-negative")
        if not 0 < self.lr_decay < 1:
            raise ValueError("lr_decay must be in (0, 1)")
        if self.min_lr <= 0:
            raise ValueError("min_lr must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
