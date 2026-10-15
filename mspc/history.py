"""Serializable rolling risk-state history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional

import torch


@dataclass
class RiskRecord:
    pre: torch.Tensor
    post: torch.Tensor


class HistoryBuffer:
    def __init__(self, maxlen: int = 5) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = int(maxlen)
        self._records: Deque[RiskRecord] = deque(maxlen=self.maxlen)
        self._module_count: Optional[int] = None

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._module_count = None

    def append(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        pre_cpu = torch.as_tensor(pre, dtype=torch.float64).detach().flatten().cpu().clone()
        post_cpu = torch.as_tensor(post, dtype=torch.float64).detach().flatten().cpu().clone()
        if pre_cpu.shape != post_cpu.shape:
            raise ValueError("pre and post risks must have the same shape")
        if not torch.isfinite(pre_cpu).all() or not torch.isfinite(post_cpu).all():
            raise ValueError("risk history cannot contain NaN or Inf")
        if torch.any((pre_cpu < 0) | (pre_cpu > 1)) or torch.any((post_cpu < 0) | (post_cpu > 1)):
            raise ValueError("normalized risks must be in [0, 1]")
        if self._module_count is None:
            self._module_count = pre_cpu.numel()
        elif pre_cpu.numel() != self._module_count:
            raise ValueError("module count changed within one history buffer")
        self._records.append(RiskRecord(pre=pre_cpu, post=post_cpu))

    def degradation(self, module_count: int, *, device=None, dtype=None) -> torch.Tensor:
        if not self._records:
            return torch.zeros(module_count, device=device, dtype=dtype or torch.float32)
        latest = self._records[-1]
        if latest.pre.numel() != module_count:
            raise ValueError("history module count does not match current risks")
        value = torch.clamp(latest.post - latest.pre, min=0.0, max=1.0)
        return value.to(device=device, dtype=dtype or torch.float32)

    def under_optimization(
        self, thresholds: torch.Tensor, *, device=None, dtype=None, eps: float = 1e-12
    ) -> torch.Tensor:
        thresholds_cpu = torch.as_tensor(thresholds, dtype=torch.float64).flatten().cpu()
        if torch.any((thresholds_cpu < 0) | (thresholds_cpu >= 1)):
            raise ValueError("risk thresholds must be in [0, 1)")
        if not self._records:
            return torch.zeros_like(thresholds_cpu, device=device, dtype=dtype or torch.float32)
        if thresholds_cpu.numel() != self._records[-1].post.numel():
            raise ValueError("threshold count does not match history")
        post_risks = torch.stack([record.post for record in self._records])
        excess = torch.clamp(post_risks - thresholds_cpu, min=0.0)
        ratios = excess / torch.clamp(1.0 - thresholds_cpu, min=eps)
        value = torch.clamp(ratios.mean(dim=0), min=0.0, max=1.0)
        return value.to(device=device, dtype=dtype or torch.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "maxlen": self.maxlen,
            "records": [
                {"pre": record.pre.clone(), "post": record.post.clone()}
                for record in self._records
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        maxlen = int(state["maxlen"])
        if maxlen <= 0:
            raise ValueError("invalid history maxlen")
        self.maxlen = maxlen
        self._records = deque(maxlen=maxlen)
        self._module_count = None
        for record in state.get("records", []):
            self.append(record["pre"], record["post"])

    def records(self) -> Iterable[RiskRecord]:
        return tuple(self._records)

