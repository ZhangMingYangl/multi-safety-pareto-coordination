"""Full checkpoints and structured diagnostic logs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .transaction import capture_rng_state, restore_rng_state
from .types import CoordinationDiagnostics


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def append_diagnostics(
    path: str | os.PathLike[str],
    diagnostics: CoordinationDiagnostics | Mapping[str, Any],
    **context: Any,
) -> None:
    """Append one UTF-8 JSON object for a coordination round."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = diagnostics.to_dict() if isinstance(diagnostics, CoordinationDiagnostics) else dict(diagnostics)
    payload = {**context, **payload}
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_value(payload), ensure_ascii=False, allow_nan=False))
        handle.write("\n")


def save_run_config(path: str | os.PathLike[str], config: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(config), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_full_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    coordinator,
    epoch: int,
    global_step: int,
    scheduler=None,
    config: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Atomically save all state required for deterministic continuation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "mspc": coordinator.state_dict(),
        "config": _json_value(config),
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def load_full_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    coordinator,
    scheduler=None,
    map_location=None,
) -> Mapping[str, Any]:
    """Restore a trusted local full checkpoint and return its metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if checkpoint["scheduler"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
    coordinator.load_state_dict(checkpoint["mspc"])
    restore_rng_state(checkpoint["rng"])
    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "config": checkpoint.get("config"),
        "extra": checkpoint.get("extra", {}),
    }

