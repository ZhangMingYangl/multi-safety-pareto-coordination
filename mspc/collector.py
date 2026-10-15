"""Gradient collection and flat-gradient writeback utilities."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch


def trainable_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    result = [parameter for parameter in parameters if parameter.requires_grad]
    if not result:
        raise ValueError("at least one trainable shared parameter is required")
    return result


def collect_gradients(
    losses: Sequence[torch.Tensor] | torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Collect every module gradient from the same parameter version.

    ``active_masks`` distinguish an unused parameter from an active parameter whose
    derivative happens to be zero.
    """
    if isinstance(losses, torch.Tensor):
        if losses.ndim == 0:
            loss_list = [losses]
        else:
            loss_list = list(losses.unbind())
    else:
        loss_list = list(losses)
    if len(loss_list) < 2:
        raise ValueError("MSPC requires at least two losses")
    params = trainable_parameters(parameters)
    total_dimension = sum(parameter.numel() for parameter in params)
    rows = []
    masks = []
    for index, loss in enumerate(loss_list):
        keep_graph = retain_graph or index < len(loss_list) - 1
        per_parameter = torch.autograd.grad(
            loss,
            params,
            retain_graph=keep_graph,
            allow_unused=True,
            materialize_grads=False,
        )
        flat_parts = []
        mask_parts = []
        for parameter, gradient in zip(params, per_parameter):
            if gradient is None:
                flat_parts.append(torch.zeros_like(parameter).reshape(-1))
                mask_parts.append(
                    torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.bool)
                )
            else:
                flat_parts.append(gradient.detach().reshape(-1))
                mask_parts.append(
                    torch.ones(parameter.numel(), device=parameter.device, dtype=torch.bool)
                )
        rows.append(torch.cat(flat_parts))
        masks.append(torch.cat(mask_parts))
    gradients = torch.stack(rows)
    active_masks = torch.stack(masks)
    if gradients.shape[1] != total_dimension:
        raise RuntimeError("collected gradient dimension is inconsistent")
    return gradients, active_masks


def write_flat_gradient(
    parameters: Iterable[torch.nn.Parameter], flat_gradient: torch.Tensor
) -> None:
    """Write one flat coordinated direction into shared ``.grad`` attributes."""
    params = trainable_parameters(parameters)
    expected = sum(parameter.numel() for parameter in params)
    if flat_gradient.ndim != 1 or flat_gradient.numel() != expected:
        raise ValueError(f"flat_gradient must have shape [{expected}]")
    if not torch.isfinite(flat_gradient).all():
        raise ValueError("refusing to write a non-finite gradient")
    offset = 0
    for parameter in params:
        end = offset + parameter.numel()
        gradient = flat_gradient[offset:end].reshape_as(parameter).detach().clone()
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.copy_(gradient)
        offset = end

