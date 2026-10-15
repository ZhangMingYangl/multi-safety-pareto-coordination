"""Post-update safety feedback and learning-rate decisions."""

from __future__ import annotations

import torch

from .types import FeedbackDecision, SafetyEvaluation


class FeedbackController:
    def __init__(
        self,
        risk_thresholds,
        utility_min: float,
        stability_min: float,
        beta: float = 0.5,
        eps: float = 1e-12,
    ) -> None:
        self.risk_thresholds = torch.as_tensor(risk_thresholds, dtype=torch.float64).flatten()
        if torch.any((self.risk_thresholds < 0) | (self.risk_thresholds >= 1)):
            raise ValueError("risk thresholds must be in [0, 1)")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.utility_min = float(utility_min)
        self.stability_min = float(stability_min)
        self.beta = float(beta)
        self.eps = float(eps)

    def assess(self, evaluation: SafetyEvaluation) -> FeedbackDecision:
        risks = torch.as_tensor(evaluation.risks, dtype=torch.float64).flatten()
        if risks.shape != self.risk_thresholds.shape:
            raise ValueError("evaluation risks do not match configured thresholds")
        if not torch.isfinite(risks).all():
            return FeedbackDecision(
                accepted=False,
                risk_excess=torch.ones_like(risks),
                risk_failed=True,
                utility_failed=True,
                stability_failed=True,
            )
        excess = torch.clamp(risks - self.risk_thresholds, min=0.0)
        excess = torch.clamp(
            excess / torch.clamp(1.0 - self.risk_thresholds, min=self.eps), 0.0, 1.0
        )
        risk_failed = bool(torch.any(excess > 0))
        utility_failed = not torch.isfinite(torch.tensor(evaluation.utility)) or (
            evaluation.utility < self.utility_min
        )
        stability_failed = not torch.isfinite(torch.tensor(evaluation.stability)) or (
            evaluation.stability < self.stability_min
        )
        return FeedbackDecision(
            accepted=not (risk_failed or utility_failed or stability_failed),
            risk_excess=excess,
            risk_failed=risk_failed,
            utility_failed=utility_failed,
            stability_failed=stability_failed,
        )

    def adjust_q(self, q: torch.Tensor, decision: FeedbackDecision) -> torch.Tensor:
        excess = decision.risk_excess.to(device=q.device, dtype=q.dtype)
        if excess.shape != q.shape:
            raise ValueError("feedback and q dimensions do not match")
        return torch.clamp(q + self.beta * excess, 0.0, 1.0)


class SafetyLrController:
    """Apply a persistent safety multiplier separately from a scheduler's base rates."""

    def __init__(self, decay: float = 0.5, min_lr: float = 1e-8) -> None:
        if not 0 < decay < 1 or min_lr <= 0:
            raise ValueError("decay must be in (0,1) and min_lr must be positive")
        self.decay = float(decay)
        self.min_lr = float(min_lr)
        self.scale = 1.0

    def decay_scale(self) -> float:
        self.scale *= self.decay
        return self.scale

    def apply(self, optimizer: torch.optim.Optimizer, base_lrs) -> None:
        if len(base_lrs) != len(optimizer.param_groups):
            raise ValueError("base_lrs must match optimizer parameter groups")
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = max(self.min_lr, float(base_lr) * self.scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state) -> None:
        self.scale = float(state["scale"])

