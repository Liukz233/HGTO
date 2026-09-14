"""Shared physical stopping rule for linear, finite-deformation and plastic HGTO."""

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StopConfig:
    window: int = 10
    objective_tolerance: float = 1e-3
    density_tolerance: float = 5e-3
    patience: int = 5
    volume_tolerance: float = 1e-7
    equilibrium_tolerance: float = 1e-8


class PhysicalStopping:
    def __init__(self, config=StopConfig()):
        if (
            config.window < 2
            or config.patience < 1
            or min(
                config.objective_tolerance,
                config.density_tolerance,
                config.volume_tolerance,
                config.equilibrium_tolerance,
            )
            <= 0
        ):
            raise ValueError("Stopping window/patience and tolerances must be positive")
        self.config = config
        self.values = deque(maxlen=config.window)
        self.previous = None
        self.key = None
        self.stable = 0

    def observe(
        self,
        objective,
        rho,
        *,
        parameters,
        continuation_complete,
        volume_error,
        residual,
        accepted=True,
    ):
        rho = np.asarray(rho, dtype=float)
        key = tuple(parameters)
        c = self.config
        valid = bool(
            np.isfinite(objective)
            and np.isfinite(rho).all()
            and np.isfinite(residual)
            and abs(volume_error) <= c.volume_tolerance
            and residual <= c.equilibrium_tolerance
            and accepted
        )
        same = self.key == key
        delta = (
            None
            if self.previous is None or not same
            else float(np.max(np.abs(rho - self.previous)))
        )
        eligible = valid and continuation_complete and same
        if not eligible:
            self.values.clear()
            self.stable = 0
        if valid and continuation_complete:
            self.values.append(float(objective))
        relative = None
        if eligible and len(self.values) == c.window:
            values = np.asarray(self.values)
            relative = float(np.ptp(values) / max(abs(values.mean()), np.finfo(float).tiny))
        satisfied = bool(
            relative is not None
            and relative < c.objective_tolerance
            and delta is not None
            and delta < c.density_tolerance
        )
        self.stable = self.stable + 1 if satisfied else 0
        self.previous = rho.copy()
        self.key = key
        return dict(
            stopping_eligible=eligible,
            objective_relative_window=relative,
            maximum_density_change=delta,
            stable_checks=self.stable,
            converged=self.stable >= c.patience,
            volume_error=float(abs(volume_error)),
            equilibrium_residual=float(residual),
            accepted_for_stopping=bool(accepted),
        )
