"""End-to-end implementation of the MSPC core coordination round."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from .balance import balance_factors
from .collector import collect_gradients, write_flat_gradient
from .config import MSPCConfig
from .conflict import (
    conflict_matrix,
    count_negative_pairs,
    normalize_gradients,
    scale_preserving_project,
    sequential_project,
)
from .history import HistoryBuffer
from .hybrid import common_descent_guard
from .pareto import ParetoWeightSolver, objective_values
from .types import CoordinationDiagnostics, CoordinationResult


class MSPCCoordinator:
    """Coordinate module gradients without directly applying an optimizer step."""

    def __init__(self, config: Optional[MSPCConfig] = None) -> None:
        self.config = config or MSPCConfig()
        self.history = HistoryBuffer(self.config.history_size)
        self.solver = ParetoWeightSolver(
            grid_step=self.config.grid_step,
            eps=self.config.eps_obj,
            sampled_candidates=self.config.sampled_candidates,
            seed=self.config.sampling_seed,
            min_weight=self.config.min_task_weight,
        )
        self.previous_alpha: Optional[torch.Tensor] = None
        self.step = 0
        self.last_diagnostics: Optional[CoordinationDiagnostics] = None

    def _history_terms(self, risks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        module_count = risks.numel()
        degradation = self.history.degradation(
            module_count, device=risks.device, dtype=risks.dtype
        )
        if self.config.risk_thresholds is None:
            under_optimization = torch.zeros_like(risks)
        else:
            thresholds = torch.as_tensor(
                self.config.risk_thresholds, device=risks.device, dtype=risks.dtype
            )
            if thresholds.shape != risks.shape:
                raise ValueError("risk_thresholds must match the module count")
            under_optimization = self.history.under_optimization(
                thresholds,
                device=risks.device,
                dtype=risks.dtype,
                eps=self.config.eps_obj,
            )
        return degradation, under_optimization

    def coordinate(
        self,
        gradients: torch.Tensor,
        risks: torch.Tensor,
        active_masks: Optional[torch.Tensor] = None,
        *,
        q_override: Optional[torch.Tensor] = None,
        commit: bool = True,
    ) -> CoordinationResult:
        risks = torch.as_tensor(risks, device=gradients.device, dtype=gradients.dtype).flatten()
        if risks.shape != (gradients.shape[0],):
            raise ValueError("risks must contain one value per module")
        if not torch.isfinite(risks).all() or torch.any((risks < 0) | (risks > 1)):
            raise ValueError("normalized risks must be finite and in [0, 1]")

        conflicts = conflict_matrix(gradients, active_masks, eps=self.config.eps_grad)
        normalized = normalize_gradients(gradients, eps=self.config.eps_grad)
        if self.config.gradient_scaling == "unit":
            projected, orders = sequential_project(
                normalized, conflicts, active_masks, eps=self.config.eps_grad
            )
        else:
            projected, orders = scale_preserving_project(
                gradients, conflicts, active_masks, eps=self.config.eps_grad
            )
        degradation, under_optimization = self._history_terms(risks)
        q = balance_factors(
            risks,
            degradation,
            under_optimization,
            (self.config.lambda_r, self.config.lambda_d, self.config.lambda_u),
        )
        if q_override is not None:
            q = torch.as_tensor(q_override, device=gradients.device, dtype=gradients.dtype)
            if q.shape != risks.shape or not torch.isfinite(q).all():
                raise ValueError("q_override must be a finite vector matching risks")
            q = torch.clamp(q, 0.0, 1.0)

        should_solve = (
            self.previous_alpha is None
            or self.step % self.config.weight_update_interval == 0
            or q_override is not None
        )
        if should_solve:
            pareto = self.solver.solve(conflicts, q, self.previous_alpha)
            selected_alpha = pareto.alpha
            alpha = selected_alpha
            if self.previous_alpha is not None and self.config.alpha_smoothing > 0:
                previous = self.previous_alpha.to(
                    device=gradients.device, dtype=gradients.dtype
                )
                alpha = (
                    self.config.alpha_smoothing * previous
                    + (1.0 - self.config.alpha_smoothing) * selected_alpha
                )
                alpha = alpha / torch.clamp(alpha.sum(), min=self.config.eps_obj)
            pareto_size = pareto.pareto_size
        else:
            alpha = self.previous_alpha.to(device=gradients.device, dtype=gradients.dtype).clone()
            selected_alpha = alpha.clone()
            # Still evaluate the reused point so failures surface immediately.
            reused_objectives = objective_values(alpha.unsqueeze(0), conflicts, q)
            if not torch.isfinite(reused_objectives).all():
                raise ValueError("reused Pareto weight produced a non-finite objective")
            pareto_size = 0
        fused = torch.sum(alpha[:, None] * projected, dim=0)
        task_dots_before = gradients @ fused
        guard_triggered = False
        guard_blend = 0.0
        mgda_weights = None
        if self.config.descent_guard == "mgda":
            guard = common_descent_guard(
                gradients, fused, tolerance=self.config.descent_tolerance
            )
            fused = guard.gradient
            task_dots_before = guard.task_dots_before
            task_dots_after = guard.task_dots_after
            guard_triggered = guard.triggered
            guard_blend = guard.blend
            mgda_weights = guard.mgda_weights.detach()
        else:
            task_dots_after = task_dots_before
        fused_scale_factor = (
            float(gradients.shape[0]) if self.config.fused_scale == "task_count" else 1.0
        )
        fused = fused * fused_scale_factor
        task_dots_before = task_dots_before * fused_scale_factor
        task_dots_after = task_dots_after * fused_scale_factor
        if not torch.isfinite(fused).all():
            raise ValueError("fused gradient contains NaN or Inf")
        pair_count = gradients.shape[0] * (gradients.shape[0] - 1) // 2
        conflict_pairs = int(torch.count_nonzero(torch.triu(conflicts, diagonal=1) > 0).item())
        diagnostics = CoordinationDiagnostics(
            grad_norms=torch.linalg.vector_norm(gradients, dim=1).detach(),
            conflict_matrix=conflicts.detach(),
            conflict_pair_rate=conflict_pairs / pair_count,
            risks=risks.detach(),
            degradation=degradation.detach(),
            under_optimization=under_optimization.detach(),
            q=q.detach(),
            alpha=alpha.detach(),
            fused_grad_norm=float(torch.linalg.vector_norm(fused).item()),
            pareto_size=pareto_size,
            projection_orders=orders,
            negative_pairs_before=count_negative_pairs(normalized),
            negative_pairs_after=count_negative_pairs(projected),
            reused_weights=not should_solve,
            gradient_scaling=self.config.gradient_scaling,
            descent_guard=self.config.descent_guard,
            guard_triggered=guard_triggered,
            guard_blend=guard_blend,
            min_task_dot_before_guard=float(task_dots_before.min().item()),
            min_task_dot_after_guard=float(task_dots_after.min().item()),
            mgda_weights=mgda_weights,
            selected_alpha=selected_alpha.detach(),
            min_task_weight=self.config.min_task_weight,
            alpha_smoothing=self.config.alpha_smoothing,
            fused_scale_factor=fused_scale_factor,
        )
        result = CoordinationResult(
            fused_gradient=fused,
            projected_gradients=projected,
            normalized_gradients=normalized,
            alpha=alpha,
            q=q,
            diagnostics=diagnostics,
        )
        if commit:
            self.previous_alpha = alpha.detach().cpu().clone()
            self.step += 1
            self.last_diagnostics = diagnostics
        return result

    def coordinate_losses(
        self,
        losses: Sequence[torch.Tensor] | torch.Tensor,
        shared_parameters: Iterable[torch.nn.Parameter],
        risks: torch.Tensor,
        *,
        retain_graph: bool = False,
        q_override: Optional[torch.Tensor] = None,
        commit: bool = True,
    ) -> CoordinationResult:
        parameters = list(shared_parameters)
        gradients, masks = collect_gradients(
            losses, parameters, retain_graph=retain_graph
        )
        result = self.coordinate(
            gradients, risks, masks, q_override=q_override, commit=commit
        )
        write_flat_gradient(parameters, result.fused_gradient)
        return result

    def record_transition(self, pre_risks: torch.Tensor, post_risks: torch.Tensor) -> None:
        self.history.append(pre_risks, post_risks)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history.state_dict(),
            "previous_alpha": None
            if self.previous_alpha is None
            else self.previous_alpha.clone(),
            "step": self.step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.history.load_state_dict(deepcopy(state["history"]))
        previous = state.get("previous_alpha")
        self.previous_alpha = None if previous is None else previous.detach().cpu().clone()
        self.step = int(state.get("step", 0))
        self.last_diagnostics = None
