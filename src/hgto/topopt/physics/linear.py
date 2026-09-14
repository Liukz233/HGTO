"""Warm-started linear-elastic compliance and physical-density gradients."""

from __future__ import annotations
import time
from typing import Any
import numpy as np
import torch
from hgto.fem import MechanicsOperator
from hgto.topopt.physics.base import Analysis


def _state_diagnostics(state: Any) -> dict[str, Any]:
    fallback = state.fallback_used
    return {
        "compliance": float(state.compliance.item()),
        "residual_rel": float(torch.max(state.residual_rel).item()),
        "state_backend": (
            "hypergraph_direct"
            if fallback is not None and bool(fallback.any().item())
            else "hypergraph"
        ),
        "solver_iterations": int(torch.sum(state.iterations).item()),
    }


class LinearCompliance:
    """Minimize ``f^T u`` under linear elasticity."""

    name = "linear_elastic"

    def __init__(
        self,
        operator: MechanicsOperator,
        force: torch.Tensor,
        *,
        rtol: float,
        max_iter: int,
        allow_direct_fallback: bool,
        bw_rtol: float,
        volumes: np.ndarray,
        volume_fraction: float,
    ) -> None:
        self.operator = operator
        self.force = force
        self.rtol = rtol
        self.max_iter = max_iter
        self.allow_direct_fallback = allow_direct_fallback
        self.bw_rtol = bw_rtol
        self.volumes = volumes
        self.volume_fraction = volume_fraction
        self._warm: torch.Tensor | None = None

    def supports(self, problem: Any) -> None:
        return None  # linear elastic compliance is the framework's base case

    def analyze(self, rho: torch.Tensor, iteration: int) -> Analysis:
        start = time.perf_counter()
        state = self.operator.solve_state(
            rho,
            self.force,
            u0=self._warm,
            rtol=self.rtol,
            max_iter=self.max_iter,
            allow_direct_fallback=self.allow_direct_fallback,
        )
        t_state = time.perf_counter() - start
        self._warm = state.u

        start = time.perf_counter()
        gradient = self.operator.compliance_sensitivity(rho, state)
        t_sens = time.perf_counter() - start

        diagnostics = _state_diagnostics(state)
        compliance = diagnostics["compliance"]
        return Analysis(
            objective=compliance,
            dJ_drho=gradient,
            normalizer_key="C0",
            normalizer_value=compliance,
            diagnostics=diagnostics,
            t_state=t_state,
            t_sens=t_sens,
        )
