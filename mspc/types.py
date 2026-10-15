"""Typed results used by the MSPC implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


def _tensor_to_list(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


@dataclass
class ParetoResult:
    alpha: torch.Tensor
    objectives: torch.Tensor
    pareto_mask: torch.BoolTensor
    selected_candidate_index: int
    ideal_distance: float

    @property
    def pareto_size(self) -> int:
        return int(self.pareto_mask.sum().item())


@dataclass
class CoordinationDiagnostics:
    grad_norms: torch.Tensor
    conflict_matrix: torch.Tensor
    conflict_pair_rate: float
    risks: torch.Tensor
    degradation: torch.Tensor
    under_optimization: torch.Tensor
    q: torch.Tensor
    alpha: torch.Tensor
    fused_grad_norm: float
    pareto_size: int
    projection_orders: List[List[int]] = field(default_factory=list)
    negative_pairs_before: int = 0
    negative_pairs_after: int = 0
    reused_weights: bool = False
    gradient_scaling: str = "unit"
    descent_guard: str = "none"
    guard_triggered: bool = False
    guard_blend: float = 0.0
    min_task_dot_before_guard: float = 0.0
    min_task_dot_after_guard: float = 0.0
    mgda_weights: Optional[torch.Tensor] = None
    selected_alpha: Optional[torch.Tensor] = None
    min_task_weight: float = 0.0
    alpha_smoothing: float = 0.0
    fused_scale_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grad_norms": _tensor_to_list(self.grad_norms),
            "conflict_matrix": _tensor_to_list(self.conflict_matrix),
            "conflict_pair_rate": self.conflict_pair_rate,
            "risks": _tensor_to_list(self.risks),
            "degradation": _tensor_to_list(self.degradation),
            "under_optimization": _tensor_to_list(self.under_optimization),
            "q": _tensor_to_list(self.q),
            "alpha": _tensor_to_list(self.alpha),
            "fused_grad_norm": self.fused_grad_norm,
            "pareto_size": self.pareto_size,
            "projection_orders": self.projection_orders,
            "negative_pairs_before": self.negative_pairs_before,
            "negative_pairs_after": self.negative_pairs_after,
            "reused_weights": self.reused_weights,
            "gradient_scaling": self.gradient_scaling,
            "descent_guard": self.descent_guard,
            "guard_triggered": self.guard_triggered,
            "guard_blend": self.guard_blend,
            "min_task_dot_before_guard": self.min_task_dot_before_guard,
            "min_task_dot_after_guard": self.min_task_dot_after_guard,
            "mgda_weights": None
            if self.mgda_weights is None
            else _tensor_to_list(self.mgda_weights),
            "selected_alpha": None
            if self.selected_alpha is None
            else _tensor_to_list(self.selected_alpha),
            "min_task_weight": self.min_task_weight,
            "alpha_smoothing": self.alpha_smoothing,
            "fused_scale_factor": self.fused_scale_factor,
        }


@dataclass
class CoordinationResult:
    fused_gradient: torch.Tensor
    projected_gradients: torch.Tensor
    normalized_gradients: torch.Tensor
    alpha: torch.Tensor
    q: torch.Tensor
    diagnostics: CoordinationDiagnostics


@dataclass(frozen=True)
class SafetyEvaluation:
    risks: torch.Tensor
    utility: float
    stability: float


@dataclass(frozen=True)
class FeedbackDecision:
    accepted: bool
    risk_excess: torch.Tensor
    risk_failed: bool
    utility_failed: bool
    stability_failed: bool


@dataclass
class FullStepResult:
    accepted: bool
    rolled_back: bool
    attempts: int
    evaluation: Optional[SafetyEvaluation]
    coordination: Optional[CoordinationResult]
    failure_reason: Optional[str] = None
