"""Full-mode transactional update, feedback retry, and exact rollback."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch

from .collector import collect_gradients, write_flat_gradient
from .coordinator import MSPCCoordinator
from .feedback import FeedbackController
from .types import FullStepResult, SafetyEvaluation


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class UpdateSnapshot:
    model: dict
    optimizer: dict
    scheduler: Optional[dict]
    coordinator: dict
    rng: dict

    @classmethod
    def capture(cls, model, optimizer, scheduler, coordinator) -> "UpdateSnapshot":
        return cls(
            model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),
            scheduler=None if scheduler is None else copy.deepcopy(scheduler.state_dict()),
            coordinator=copy.deepcopy(coordinator.state_dict()),
            rng=capture_rng_state(),
        )

    def restore(self, model, optimizer, scheduler, coordinator) -> None:
        model.load_state_dict(copy.deepcopy(self.model))
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        if scheduler is not None:
            if self.scheduler is None:
                raise ValueError("snapshot has no scheduler state")
            scheduler.load_state_dict(copy.deepcopy(self.scheduler))
        coordinator.load_state_dict(copy.deepcopy(self.coordinator))
        restore_rng_state(self.rng)


class FullMSPCRunner:
    """Run one feedback-validated optimizer transaction.

    ``compute_losses`` must recompute all losses from the current parameter version.
    ``evaluate`` must use validation data independent of the final test set.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        coordinator: MSPCCoordinator,
        feedback: FeedbackController,
        *,
        scheduler=None,
        max_retries: int = 1,
        lr_decay: float = 0.5,
        min_lr: float = 1e-8,
    ) -> None:
        if max_retries < 0 or not 0 < lr_decay < 1 or min_lr <= 0:
            raise ValueError("invalid retry or learning-rate settings")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.coordinator = coordinator
        self.feedback = feedback
        self.max_retries = max_retries
        self.lr_decay = lr_decay
        self.min_lr = min_lr

    def run_step(
        self,
        compute_losses: Callable[[], Sequence[torch.Tensor] | torch.Tensor],
        shared_parameters: Iterable[torch.nn.Parameter],
        pre_risks: torch.Tensor,
        evaluate: Callable[[], SafetyEvaluation],
    ) -> FullStepResult:
        parameters = list(shared_parameters)
        snapshot = UpdateSnapshot.capture(
            self.model, self.optimizer, self.scheduler, self.coordinator
        )
        base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        q_override = None
        lr_scale = 1.0
        last_evaluation = None
        last_coordination = None
        last_reason = None

        try:
            for attempt in range(self.max_retries + 1):
                if attempt > 0:
                    snapshot.restore(
                        self.model, self.optimizer, self.scheduler, self.coordinator
                    )
                for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    group["lr"] = max(self.min_lr, base_lr * lr_scale)
                self.optimizer.zero_grad(set_to_none=True)
                losses = compute_losses()
                gradients, masks = collect_gradients(losses, parameters)
                last_coordination = self.coordinator.coordinate(
                    gradients,
                    pre_risks,
                    masks,
                    q_override=q_override,
                    commit=True,
                )
                write_flat_gradient(parameters, last_coordination.fused_gradient)
                self.optimizer.step()
                last_evaluation = evaluate()
                decision = self.feedback.assess(last_evaluation)
                if decision.accepted:
                    self.coordinator.record_transition(pre_risks, last_evaluation.risks)
                    return FullStepResult(
                        accepted=True,
                        rolled_back=False,
                        attempts=attempt + 1,
                        evaluation=last_evaluation,
                        coordination=last_coordination,
                    )
                reasons = []
                if decision.risk_failed:
                    q_override = self.feedback.adjust_q(last_coordination.q, decision)
                    reasons.append("risk")
                if decision.utility_failed:
                    reasons.append("utility")
                if decision.stability_failed:
                    reasons.append("stability")
                if decision.utility_failed or decision.stability_failed:
                    lr_scale *= self.lr_decay
                last_reason = ",".join(reasons)
        except Exception:
            # Programming and numerical errors must never leave a partial Adam update behind.
            snapshot.restore(self.model, self.optimizer, self.scheduler, self.coordinator)
            raise

        snapshot.restore(self.model, self.optimizer, self.scheduler, self.coordinator)
        return FullStepResult(
            accepted=False,
            rolled_back=True,
            attempts=self.max_retries + 1,
            evaluation=last_evaluation,
            coordination=last_coordination,
            failure_reason=last_reason,
        )
