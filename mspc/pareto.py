"""Deterministic simplex candidates and Pareto ideal-point selection."""

from __future__ import annotations

from typing import Optional

import torch

from .types import ParetoResult


def simplex_grid(
    module_count: int,
    step: float = 0.05,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if module_count < 2:
        raise ValueError("module_count must be at least two")
    units_float = 1.0 / step
    units = int(round(units_float))
    if abs(units_float - units) > 1e-8:
        raise ValueError("step must divide one exactly")

    rows = []

    def visit(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, dimensions - 1)

    visit([], units, module_count)
    return torch.tensor(rows, device=device, dtype=dtype) / float(units)


def _deduplicate_rows(rows: torch.Tensor, decimals: int = 12) -> torch.Tensor:
    scale = float(10**decimals)
    keys = torch.round(rows.detach().cpu() * scale).to(torch.int64).tolist()
    seen = set()
    keep = []
    for index, key in enumerate(keys):
        key_tuple = tuple(key)
        if key_tuple not in seen:
            seen.add(key_tuple)
            keep.append(index)
    return rows[torch.tensor(keep, device=rows.device)]


def generate_candidates(
    module_count: int,
    step: float = 0.05,
    previous_alpha: Optional[torch.Tensor] = None,
    *,
    sampled_candidates: int = 2048,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    min_weight: float = 0.0,
) -> torch.Tensor:
    """Use a grid for up to three modules and seeded Dirichlet sampling above that."""
    if min_weight < 0 or module_count * min_weight >= 1:
        raise ValueError("min_weight must be non-negative and module_count * min_weight < 1")
    if module_count <= 3:
        candidates = simplex_grid(module_count, step, device=device, dtype=dtype)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        exponential = -torch.log(
            torch.rand(sampled_candidates, module_count, generator=generator).clamp_min(1e-12)
        )
        candidates = (exponential / exponential.sum(dim=1, keepdim=True)).to(
            device=device, dtype=dtype
        )
    vertices = torch.eye(module_count, device=device, dtype=dtype)
    equal = torch.full((1, module_count), 1.0 / module_count, device=device, dtype=dtype)
    candidates = torch.cat([candidates, vertices, equal], dim=0)
    if min_weight > 0:
        candidates = min_weight + (1.0 - module_count * min_weight) * candidates
    extras = [candidates]
    if previous_alpha is not None:
        previous = torch.as_tensor(previous_alpha, device=device, dtype=dtype).reshape(1, -1)
        if previous.shape[1] != module_count:
            raise ValueError("previous_alpha has the wrong module count")
        if torch.any(previous < 0) or abs(float(previous.sum().item()) - 1.0) > 1e-6:
            raise ValueError("previous_alpha must lie on the probability simplex")
        if min_weight > 0 and torch.any(previous < min_weight - 1e-7):
            # Deterministic migration for state created before a floor was enabled.
            previous = min_weight + (1.0 - module_count * min_weight) * previous
        extras.append(previous)
    return _deduplicate_rows(torch.cat(extras, dim=0))


def objective_values(
    candidates: torch.Tensor, conflicts: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2-D tensor")
    module_count = candidates.shape[1]
    if conflicts.shape != (module_count, module_count) or q.shape != (module_count,):
        raise ValueError("objective input dimensions do not match")
    f1 = 0.5 * torch.sum((candidates @ conflicts) * candidates, dim=1)
    f2 = -(candidates @ q)
    f3 = torch.sum((candidates - 1.0 / module_count) ** 2, dim=1)
    values = torch.stack((f1, f2, f3), dim=1)
    if not torch.isfinite(values).all():
        raise ValueError("Pareto objectives contain NaN or Inf")
    return values


def nondominated_mask(objectives: torch.Tensor, atol: float = 1e-12) -> torch.BoolTensor:
    """Return candidates not dominated by another row for all-minimization objectives."""
    if objectives.ndim != 2:
        raise ValueError("objectives must be a 2-D tensor")
    lhs = objectives[:, None, :]
    rhs = objectives[None, :, :]
    dominates = torch.all(lhs <= rhs + atol, dim=2) & torch.any(lhs < rhs - atol, dim=2)
    return ~torch.any(dominates, dim=0)


def select_ideal(
    candidates: torch.Tensor,
    objectives: torch.Tensor,
    pareto_mask: torch.BoolTensor,
    eps: float = 1e-12,
) -> tuple[int, float]:
    pareto_indices = torch.nonzero(pareto_mask, as_tuple=False).flatten()
    if pareto_indices.numel() == 0:
        raise ValueError("Pareto set is empty")
    pareto_objectives = objectives[pareto_indices]
    minimum = pareto_objectives.min(dim=0).values
    maximum = pareto_objectives.max(dim=0).values
    normalized = (pareto_objectives - minimum) / (maximum - minimum + eps)
    distances = torch.linalg.vector_norm(normalized, dim=1)

    # Explicit CPU tuple ordering makes all documented tie breakers deterministic.
    sortable = []
    for local_index, candidate_index in enumerate(pareto_indices.tolist()):
        alpha_key = tuple(float(value) for value in candidates[candidate_index].detach().cpu())
        sortable.append(
            (
                float(distances[local_index].item()),
                float(objectives[candidate_index, 0].item()),
                float(objectives[candidate_index, 2].item()),
                alpha_key,
                candidate_index,
            )
        )
    selected = min(sortable)
    return int(selected[-1]), float(selected[0])


class ParetoWeightSolver:
    def __init__(
        self,
        grid_step: float = 0.05,
        eps: float = 1e-12,
        sampled_candidates: int = 2048,
        seed: int = 0,
        min_weight: float = 0.0,
    ) -> None:
        self.grid_step = grid_step
        self.eps = eps
        self.sampled_candidates = sampled_candidates
        self.seed = seed
        self.min_weight = min_weight

    def solve(
        self,
        conflicts: torch.Tensor,
        q: torch.Tensor,
        previous_alpha: Optional[torch.Tensor] = None,
    ) -> ParetoResult:
        module_count = q.numel()
        candidates = generate_candidates(
            module_count,
            self.grid_step,
            previous_alpha,
            sampled_candidates=self.sampled_candidates,
            seed=self.seed,
            device=q.device,
            dtype=q.dtype,
            min_weight=self.min_weight,
        )
        objectives = objective_values(candidates, conflicts, q)
        mask = nondominated_mask(objectives, atol=self.eps)
        selected_index, distance = select_ideal(candidates, objectives, mask, eps=self.eps)
        return ParetoResult(
            alpha=candidates[selected_index].clone(),
            objectives=objectives,
            pareto_mask=mask,
            selected_candidate_index=selected_index,
            ideal_distance=distance,
        )
