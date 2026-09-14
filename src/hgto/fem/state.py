"""Displacements, energies and convergence information from a mechanics solve."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class MechanicsState:
    """Fields produced by a state or adjoint solve, with per-load certificates.

    Certificates (``residual_rel``, ``iterations``, ``converged``) are
    one-dimensional tensors of length ``L`` (the load-batch axis, ).
    Every tensor is detached in ``__post_init__`` so state solves never retain
    an unrolled autograd graph — this dataclass is the single
    enforcement point of that seam.
    """

    u: torch.Tensor
    strain_gp: torch.Tensor
    stress_gp: torch.Tensor
    unit_strain_energy_gp: torch.Tensor  # UNHALVED eps:C0:eps
    strain_energy_gp: torch.Tensor  # physical 1/2-form
    f_int: torch.Tensor
    residual_rel: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    compliance: torch.Tensor

    # Per-load flag: the certified direct rescue replaced a stalled iterative
    # solve. None when the solve policy disallows fallback.
    fallback_used: torch.Tensor | None = None

    # Nonlinear (NH) extension fields — None on the linear-elastic path.
    F_gp: torch.Tensor | None = None
    detF_min: torch.Tensor | None = None
    newton_iterations: torch.Tensor | None = None
    backtracks: torch.Tensor | None = None
    pcg_iterations: torch.Tensor | None = None
    load_factors: torch.Tensor | None = None
    indefinite_fallbacks: torch.Tensor | None = None
    strain_energy: torch.Tensor | None = None
    potential: torch.Tensor | None = None
    solver_rtol: torch.Tensor | None = None
    # Inner directions accepted at a stalled (above-target) MINRES residual
    # under the inexact-Newton policy — descent + Armijo were the gates.
    inexact_directions: torch.Tensor | None = None
    # Per-(load, ramp-step) NH diagnostics (E8 regime-map material).
    ramp_newton_iterations: torch.Tensor | None = None
    ramp_residual_rel: torch.Tensor | None = None
    # Converged displacement per ramp step, shape (L, n_ramp, n_nodes, 2).
    # Populated only when the solve is called with record_ramp_history=True
    # (X2 F-d exhibits); None otherwise so default solves carry no extra state.
    ramp_u: torch.Tensor | None = None
    t_state_total: float = 0.0

    def __post_init__(self) -> None:
        for state_field in fields(self):
            value = getattr(self, state_field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, state_field.name, value.detach())
