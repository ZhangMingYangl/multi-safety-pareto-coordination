"""Gradient conflict measurement, normalization, and projection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def _validate_gradients(
    gradients: torch.Tensor, active_masks: Optional[torch.Tensor]
) -> torch.BoolTensor:
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [module_count, gradient_dim]")
    if gradients.shape[0] < 2:
        raise ValueError("MSPC requires at least two modules")
    if not gradients.is_floating_point():
        raise TypeError("gradients must be floating point")
    if not torch.isfinite(gradients).all():
        raise ValueError("gradients contain NaN or Inf")
    if active_masks is None:
        return torch.ones_like(gradients, dtype=torch.bool)
    if active_masks.shape != gradients.shape:
        raise ValueError("active_masks must have the same shape as gradients")
    return active_masks.to(device=gradients.device, dtype=torch.bool)


def conflict_matrix(
    gradients: torch.Tensor,
    active_masks: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the symmetric negative-cosine conflict matrix on pairwise intersections."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    masks = _validate_gradients(gradients, active_masks)
    module_count = gradients.shape[0]
    result = gradients.new_zeros((module_count, module_count))
    for i in range(module_count):
        for j in range(i + 1, module_count):
            common = masks[i] & masks[j]
            if not bool(common.any()):
                continue
            grad_i = gradients[i, common]
            grad_j = gradients[j, common]
            norm_i = torch.linalg.vector_norm(grad_i)
            norm_j = torch.linalg.vector_norm(grad_j)
            if norm_i <= eps or norm_j <= eps:
                continue
            denominator = torch.clamp(norm_i * norm_j, min=eps)
            conflict = torch.clamp(-torch.dot(grad_i, grad_j) / denominator, 0.0, 1.0)
            result[i, j] = conflict
            result[j, i] = conflict
    return result


def normalize_gradients(gradients: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each complete shared gradient without restoring its original norm."""
    _validate_gradients(gradients, None)
    if eps <= 0:
        raise ValueError("eps must be positive")
    norms = torch.linalg.vector_norm(gradients, dim=1, keepdim=True)
    normalized = gradients / (norms + eps)
    if not torch.isfinite(normalized).all():
        raise ValueError("gradient normalization produced NaN or Inf")
    return normalized


def sequential_project(
    normalized_gradients: torch.Tensor,
    conflicts: torch.Tensor,
    active_masks: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """Project conflicts in descending strength, breaking ties by module index."""
    masks = _validate_gradients(normalized_gradients, active_masks)
    module_count = normalized_gradients.shape[0]
    if conflicts.shape != (module_count, module_count):
        raise ValueError("conflicts must have shape [module_count, module_count]")
    if not torch.isfinite(conflicts).all():
        raise ValueError("conflicts contain NaN or Inf")

    projected = normalized_gradients.clone()
    orders: List[List[int]] = []
    for i in range(module_count):
        candidates = [j for j in range(module_count) if j != i and conflicts[i, j] > 0]
        candidates.sort(key=lambda j: (-float(conflicts[i, j].item()), j))
        orders.append(candidates)
        for j in candidates:
            common = masks[i] & masks[j]
            if not bool(common.any()):
                continue
            reference = normalized_gradients[j] * common.to(normalized_gradients.dtype)
            reference_norm_sq = torch.dot(reference, reference)
            if reference_norm_sq <= eps:
                continue
            dot = torch.dot(projected[i], reference)
            if dot < 0:
                projected[i] = projected[i] - dot * reference / (reference_norm_sq + eps)
    if not torch.isfinite(projected).all():
        raise ValueError("conflict projection produced NaN or Inf")
    return projected, orders


def scale_preserving_project(
    gradients: torch.Tensor,
    conflicts: torch.Tensor,
    active_masks: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """Project raw gradients against unit references while retaining target norms.

    This engineering variant follows the scale-preserving idea used by the official
    PCGrad/CAGrad NYUv2 implementations.  MSPC's deterministic conflict-descending
    order and active-parameter intersections are retained.
    """
    masks = _validate_gradients(gradients, active_masks)
    module_count = gradients.shape[0]
    if conflicts.shape != (module_count, module_count):
        raise ValueError("conflicts must have shape [module_count, module_count]")
    if not torch.isfinite(conflicts).all():
        raise ValueError("conflicts contain NaN or Inf")

    references = normalize_gradients(gradients, eps=eps)
    projected = gradients.clone()
    orders: List[List[int]] = []
    for i in range(module_count):
        candidates = [j for j in range(module_count) if j != i and conflicts[i, j] > 0]
        candidates.sort(key=lambda j: (-float(conflicts[i, j].item()), j))
        orders.append(candidates)
        for j in candidates:
            common = masks[i] & masks[j]
            if not bool(common.any()):
                continue
            reference = references[j] * common.to(gradients.dtype)
            reference_norm_sq = torch.dot(reference, reference)
            if reference_norm_sq <= eps:
                continue
            dot = torch.dot(projected[i], reference)
            if dot < 0:
                projected[i] = projected[i] - dot * reference / (reference_norm_sq + eps)
    if not torch.isfinite(projected).all():
        raise ValueError("scale-preserving projection produced NaN or Inf")
    return projected, orders


def count_negative_pairs(gradients: torch.Tensor, tolerance: float = 1e-7) -> int:
    """Count unordered gradient pairs whose full-space dot product is negative."""
    count = 0
    for i in range(gradients.shape[0]):
        for j in range(i + 1, gradients.shape[0]):
            if torch.dot(gradients[i], gradients[j]) < -tolerance:
                count += 1
    return count
