"""LibMTL ``AbsWeighting`` adapter for MSPC core mode."""

from __future__ import annotations

import warnings

import torch

try:
    from LibMTL.weighting.abstract_weighting import AbsWeighting
except ModuleNotFoundError as error:  # pragma: no cover - depends on optional framework
    raise ModuleNotFoundError(
        "mspc.libmtl requires LibMTL to be installed or present on PYTHONPATH"
    ) from error

from .config import MSPCConfig
from .coordinator import MSPCCoordinator
from .persistence import append_diagnostics


class MSPC(AbsWeighting):
    """Drop-in LibMTL weighting class for shared-parameter gradients."""

    def __init__(self):
        super().__init__()

    def init_param(self):
        # LibMTL calls init_param before weight_args reach backward, so construction is lazy.
        self._mspc_coordinator = None
        self.last_mspc_diagnostics = None
        self._mspc_warned_default_bounds = False

    def _ensure_coordinator(self, kwargs) -> MSPCCoordinator:
        if self._mspc_coordinator is None:
            thresholds = kwargs.get("mspc_risk_thresholds")
            config = MSPCConfig(
                mode=kwargs.get("mspc_mode", "core"),
                eps_grad=float(kwargs.get("mspc_eps_grad", 1e-8)),
                eps_obj=float(kwargs.get("mspc_eps_obj", 1e-12)),
                lambda_r=float(kwargs.get("mspc_lambda_r", 1 / 3)),
                lambda_d=float(kwargs.get("mspc_lambda_d", 1 / 3)),
                lambda_u=float(kwargs.get("mspc_lambda_u", 1 / 3)),
                history_size=int(kwargs.get("mspc_history_size", 5)),
                grid_step=float(kwargs.get("mspc_grid_step", 0.05)),
                weight_update_interval=int(kwargs.get("mspc_weight_update_interval", 1)),
                risk_thresholds=thresholds,
                sampled_candidates=int(kwargs.get("mspc_sampled_candidates", 2048)),
                sampling_seed=int(kwargs.get("mspc_sampling_seed", 0)),
                gradient_scaling=kwargs.get("mspc_gradient_scaling", "unit"),
                descent_guard=kwargs.get("mspc_descent_guard", "none"),
                descent_tolerance=float(kwargs.get("mspc_descent_tolerance", 1e-7)),
                min_task_weight=float(kwargs.get("mspc_min_task_weight", 0.0)),
                alpha_smoothing=float(kwargs.get("mspc_alpha_smoothing", 0.0)),
                fused_scale=kwargs.get("mspc_fused_scale", "unit"),
            )
            self._mspc_coordinator = MSPCCoordinator(config)
        return self._mspc_coordinator

    def _risks_from_losses(self, losses, kwargs):
        explicit = kwargs.get("mspc_risks")
        if explicit is not None:
            return torch.as_tensor(explicit, device=losses.device, dtype=losses.dtype)
        if (
            "mspc_risk_lower" not in kwargs
            and "mspc_risk_upper" not in kwargs
            and not self._mspc_warned_default_bounds
        ):
            warnings.warn(
                "MSPC is using fallback loss-risk bounds [0, 1]. Calibrate and freeze "
                "--mspc_risk_lower/upper before reporting formal experiments.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._mspc_warned_default_bounds = True
        lower = torch.as_tensor(
            kwargs.get("mspc_risk_lower", [0.0] * self.task_num),
            device=losses.device,
            dtype=losses.dtype,
        )
        upper = torch.as_tensor(
            kwargs.get("mspc_risk_upper", [1.0] * self.task_num),
            device=losses.device,
            dtype=losses.dtype,
        )
        if lower.shape != (self.task_num,) or upper.shape != (self.task_num,):
            raise ValueError("mspc_risk_lower/upper must have one value per task")
        span = upper - lower
        if torch.any(span <= 0):
            raise ValueError("each mspc_risk_upper must exceed its lower bound")
        return torch.clamp((losses.detach() - lower) / span, 0.0, 1.0)

    def backward(self, losses, **kwargs):
        if self.rep_grad:
            raise ValueError("MSPC requires rep_grad=False")
        if kwargs.get("mspc_mode", "core") != "core":
            raise ValueError("LibMTL weighting adapter handles core mode; use FullMSPCRunner for full mode")
        self._compute_grad_dim()
        gradients = self._compute_grad(losses, mode="backward")
        risks = self._risks_from_losses(losses, kwargs)
        coordinator = self._ensure_coordinator(kwargs)
        result = coordinator.coordinate(gradients, risks)
        self._reset_grad(result.fused_gradient)
        self.last_mspc_diagnostics = result.diagnostics.to_dict()
        diagnostics_path = kwargs.get("mspc_diagnostics_path")
        log_interval = int(kwargs.get("mspc_log_interval", 1))
        if log_interval <= 0:
            raise ValueError("mspc_log_interval must be positive")
        if diagnostics_path and coordinator.step % log_interval == 0:
            append_diagnostics(
                diagnostics_path,
                result.diagnostics,
                coordination_step=coordinator.step,
            )
        return result.alpha.detach().cpu().numpy()

    def mspc_state_dict(self):
        if self._mspc_coordinator is None:
            return None
        return self._mspc_coordinator.state_dict()

    def load_mspc_state_dict(self, state):
        if state is None:
            return
        if self._mspc_coordinator is None:
            self._mspc_coordinator = MSPCCoordinator()
        self._mspc_coordinator.load_state_dict(state)
