"""Risk/degradation/history balance-factor computation."""

from __future__ import annotations

from typing import Tuple

import torch


def balance_factors(
    risks: torch.Tensor,
    degradation: torch.Tensor,
    under_optimization: torch.Tensor,
    lambdas: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> torch.Tensor:
    risks = torch.as_tensor(risks)
    degradation = torch.as_tensor(degradation, device=risks.device, dtype=risks.dtype)
    under_optimization = torch.as_tensor(
        under_optimization, device=risks.device, dtype=risks.dtype
    )
    if risks.ndim != 1 or degradation.shape != risks.shape or under_optimization.shape != risks.shape:
        raise ValueError("risks, degradation, and under_optimization must be equal 1-D vectors")
    if not all(torch.isfinite(value).all() for value in (risks, degradation, under_optimization)):
        raise ValueError("balance inputs contain NaN or Inf")
    if any(value < 0 for value in lambdas) or abs(sum(lambdas) - 1.0) > 1e-8:
        raise ValueError("lambdas must be non-negative and sum to one")
    q = lambdas[0] * risks + lambdas[1] * degradation + lambdas[2] * under_optimization
    return torch.clamp(q, min=0.0, max=1.0)

