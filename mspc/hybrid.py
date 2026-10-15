"""Mature multi-task optimization components used by the MSPC hybrid variant.

The minimum-norm routine is a modern PyTorch adaptation of the MGDA solver from
Ozan Sener and Vladlen Koltun's official MIT-licensed implementation.  See
THIRD_PARTY_NOTICES.md for provenance and license details.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _two_point_solution(v1v1: float, v1v2: float, v2v2: float) -> tuple[float, float]:
    if v1v2 >= v1v1:
        return 0.999, v1v1
    if v1v2 >= v2v2:
        return 0.001, v2v2
    denominator = max(v1v1 + v2v2 - 2.0 * v1v2, 1e-30)
    gamma = -(v1v2 - v2v2) / denominator
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def _project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""
    sorted_values = torch.sort(values, descending=True).values
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(
        1, values.numel() + 1, device=values.device, dtype=values.dtype
    )
    condition = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(condition, as_tuple=False)[-1].item())
    threshold = cumulative[rho] / indices[rho]
    return torch.clamp(values - threshold, min=0.0)


def min_norm_weights(
    gradients: torch.Tensor,
    *,
    max_iter: int = 250,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Find the minimum-norm convex combination of task gradients (MGDA)."""
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must have shape [task_count, gradient_dim]")
    if not gradients.is_floating_point() or not torch.isfinite(gradients).all():
        raise ValueError("gradients must be finite floating point values")
    if max_iter <= 0 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")

    gram = gradients @ gradients.t()
    task_count = gradients.shape[0]
    best_pair: tuple[int, int] | None = None
    best_gamma = 0.5
    best_cost = float("inf")
    for i in range(task_count):
        for j in range(i + 1, task_count):
            gamma, cost = _two_point_solution(
                float(gram[i, i].item()),
                float(gram[i, j].item()),
                float(gram[j, j].item()),
            )
            if cost < best_cost:
                best_pair, best_gamma, best_cost = (i, j), gamma, cost
    if best_pair is None:
        raise RuntimeError("failed to initialize minimum-norm solver")

    weights = gradients.new_zeros(task_count)
    weights[best_pair[0]] = best_gamma
    weights[best_pair[1]] = 1.0 - best_gamma
    if task_count == 2:
        return weights

    for _ in range(max_iter):
        direction = -(gram @ weights)
        candidate = _project_simplex(weights + direction)
        v1v1 = float((weights @ gram @ weights).item())
        v1v2 = float((weights @ gram @ candidate).item())
        v2v2 = float((candidate @ gram @ candidate).item())
        gamma, _ = _two_point_solution(v1v1, v1v2, v2v2)
        updated = gamma * weights + (1.0 - gamma) * candidate
        if torch.sum(torch.abs(updated - weights)) < tolerance:
            weights = updated
            break
        weights = updated
    weights = torch.clamp(weights, min=0.0)
    return weights / torch.clamp(weights.sum(), min=torch.finfo(weights.dtype).eps)


@dataclass(frozen=True)
class DescentGuardResult:
    gradient: torch.Tensor
    triggered: bool
    blend: float
    task_dots_before: torch.Tensor
    task_dots_after: torch.Tensor
    mgda_weights: torch.Tensor


def common_descent_guard(
    gradients: torch.Tensor,
    candidate: torch.Tensor,
    *,
    tolerance: float = 1e-7,
) -> DescentGuardResult:
    """Blend minimally toward MGDA when the candidate increases a task locally.

    For an optimizer update ``theta <- theta - lr * direction``, every task with
    ``dot(g_i, direction) >= 0`` is a first-order non-increase.  The routine keeps
    the patent-derived candidate whenever that condition already holds.
    """
    if candidate.ndim != 1 or candidate.shape[0] != gradients.shape[1]:
        raise ValueError("candidate must be a vector matching gradient_dim")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    weights = min_norm_weights(gradients)
    safe_direction = torch.sum(weights[:, None] * gradients, dim=0)
    before = gradients @ candidate
    safe_dots = gradients @ safe_direction
    if bool(torch.all(before >= -tolerance)):
        return DescentGuardResult(candidate, False, 0.0, before, before, weights)

    blend = 0.0
    for index in torch.nonzero(before < -tolerance, as_tuple=False).flatten().tolist():
        start = float(before[index].item())
        end = float(safe_dots[index].item())
        if end <= start + 1e-15:
            blend = 1.0
            break
        blend = max(blend, min(1.0, (-tolerance - start) / (end - start)))
    guarded = (1.0 - blend) * candidate + blend * safe_direction
    after = gradients @ guarded
    if bool(torch.any(after < -2.0 * tolerance)):
        guarded, after, blend = safe_direction, safe_dots, 1.0
    return DescentGuardResult(guarded, True, float(blend), before, after, weights)
