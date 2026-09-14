"""Objective and sensitivity returned by the physics branch."""

from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass(frozen=True)
class Analysis:
    objective: float
    dJ_drho: torch.Tensor
    normalizer_key: str = "C0"
    normalizer_value: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    t_state: float = 0.0
    t_sens: float = 0.0
