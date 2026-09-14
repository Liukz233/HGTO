"""Implicit density sensitivities for Wang-interpolated Neo-Hookean mechanics.

The converged state supplies the tangent. A transpose tangent solve and
analytic density derivatives form the compliance sensitivity without
unrolling Newton iterations. An explicit linear_solver callback may
replace the default tangent solver for both single and multiple loads."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from hgto.fem.physics.neohookean.constitutive import (
    deformation_gradient,
    nh_tangent_diagonal,
    nh_tangent_matvec,
    wang2014_energy_rho_derivative,
    wang2014_first_piola_rho_derivative,
)
from hgto.fem.physics.neohookean.newton import (
    BETA0,
    ETA0,
    GAMMA_MODE,
    GAMMA_Q,
    _wang_kwargs,
    nh_lame_parameters,
    solve_reduced_tangent,
    wang2014_tangent_at,
)
from hgto.fem.state import MechanicsState


@torch.no_grad()
def solve_wang2014_adjoint(
    operator,
    rho: torch.Tensor,
    state: MechanicsState,
    rhs: torch.Tensor,
    rtol: float = 1.0e-10,
    max_iter: int = 20000,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
    linear_solver: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the transpose tangent system at the converged state.

    Returns (adjoint fields (L, Nn, 2), per-load residual certificates,
    per-load inner iteration counts). The PCG -> MINRES strategy matches the
    state solve — an indefinite converged tangent is handled, not fatal.
    """
    operator._check_rho(rho)
    operator._check_vector_field(rhs, "rhs")
    operator._check_vector_field(state.u, "state.u")
    if rhs.shape[0] != state.u.shape[0]:
        raise ValueError("state and rhs load counts differ")
    tangent, _, invalid = wang2014_tangent_at(
        operator, state.u, rho, gamma_mode=gamma_mode, q=q, beta0=beta0, eta0=eta0
    )
    if bool(torch.any(invalid).item()):
        raise RuntimeError("cannot solve a Wang2014 adjoint at invalid F_gamma")
    diagonal = nh_tangent_diagonal(
        tangent,
        operator.econn,
        operator.dN_dx,
        operator.integration_weights,
        operator.n_nodes,
    )
    solutions, residuals, iteration_counts = [], [], []
    for load_index in range(rhs.shape[0]):
        rhs_reduced = rhs[load_index].reshape(-1)[operator.free_dof_mask]
        diagonal_reduced = diagonal[load_index, operator.free_dof_mask]

        def matvec(x: torch.Tensor, index: int = load_index) -> torch.Tensor:
            full = x.new_zeros(operator.n_dof)
            full[operator.free_dof_mask] = x
            product = nh_tangent_matvec(
                tangent[index : index + 1],
                operator._masked(full.reshape(1, operator.n_nodes, 2)),
                operator.econn,
                operator.dN_dx,
                operator.integration_weights,
                n_nodes=operator.n_nodes,
                transpose=True,
            )
            return operator._masked(product).reshape(-1)[operator.free_dof_mask]

        if linear_solver is None:
            reduced, count, _fallbacks, converged = solve_reduced_tangent(
                matvec, rhs_reduced, diagonal_reduced, rtol, rtol, max_iter
            )
        else:
            element_matrices = torch.einsum(
                "egiJkL,egaJ,egbL,eg->eaibk",
                tangent[load_index],
                operator.dN_dx,
                operator.dN_dx,
                operator.integration_weights,
            ).reshape(operator.n_elements, 8, 8)
            reduced, _, count, converged = linear_solver(
                operator,
                element_matrices,
                rhs_reduced,
                matvec,
                rtol,
                transpose=True,
            )
        residual_rel = torch.linalg.vector_norm(
            rhs_reduced - matvec(reduced)
        ) / torch.linalg.vector_norm(rhs_reduced).clamp_min(torch.finfo(rhs_reduced.dtype).tiny)
        if not converged:
            raise RuntimeError(
                "Wang2014 adjoint solve failed: residual={:.6e}, iterations={}".format(
                    float(residual_rel.item()), count
                )
            )
        full = reduced.new_zeros(operator.n_dof)
        full[operator.free_dof_mask] = reduced
        solutions.append(full.reshape(operator.n_nodes, 2))
        residuals.append(residual_rel)
        iteration_counts.append(count)
    return (
        torch.stack(solutions, dim=0),
        torch.stack(residuals, dim=0),
        torch.tensor(iteration_counts, dtype=torch.long, device=operator.device),
    )


@torch.no_grad()
def wang2014_vjp_rho(
    operator,
    rho: torch.Tensor,
    state: MechanicsState,
    adjoint: torch.Tensor,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
) -> torch.Tensor:
    """Return ``adjoint^T dR/drho`` including BOTH density paths (s', gamma')."""
    operator._check_rho(rho)
    operator._check_vector_field(state.u, "state.u")
    operator._check_vector_field(adjoint, "adjoint")
    if state.u.shape[0] != adjoint.shape[0]:
        raise ValueError("state and adjoint load counts differ")
    F = deformation_gradient(operator._masked(state.u), operator.econn, operator.dN_dx)
    mu, lam = nh_lame_parameters(operator)
    stress_rho, invalid = wang2014_first_piola_rho_derivative(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_invalid=True,
        **_wang_kwargs(gamma_mode, q, beta0, eta0),
    )
    if bool(torch.any(invalid).item()):
        raise RuntimeError("cannot evaluate Wang2014 sensitivity at invalid F_gamma")
    element_force_rho = torch.einsum(
        "legiJ,egaJ,eg->leai",
        stress_rho,
        operator.dN_dx,
        operator.integration_weights,
    )
    element_adjoint = operator._masked(adjoint)[:, operator.econn, :]
    return torch.sum(element_force_rho * element_adjoint, dim=(0, 2, 3))


@torch.no_grad()
def wang2014_compliance_sensitivity(
    operator,
    rho: torch.Tensor,
    state: MechanicsState,
    f_ext: torch.Tensor,
    rtol: float = 1.0e-10,
    max_iter: int = 20000,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
    linear_solver: Optional[Callable] = None,
) -> torch.Tensor:
    """End-compliance sensitivity ``dC/drho = -lambda^T R_rho`` (analytic)."""
    adjoint, _, _ = solve_wang2014_adjoint(
        operator,
        rho,
        state,
        f_ext,
        rtol=rtol,
        max_iter=max_iter,
        gamma_mode=gamma_mode,
        q=q,
        beta0=beta0,
        eta0=eta0,
        linear_solver=linear_solver,
    )
    return -wang2014_vjp_rho(
        operator,
        rho,
        state,
        adjoint,
        gamma_mode=gamma_mode,
        q=q,
        beta0=beta0,
        eta0=eta0,
    )


@torch.no_grad()
def wang2014_potential_sensitivity(
    operator,
    rho: torch.Tensor,
    state: MechanicsState,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
) -> torch.Tensor:
    """Equilibrium potential sensitivity from explicit energy rho-derivatives."""
    operator._check_rho(rho)
    F = deformation_gradient(operator._masked(state.u), operator.econn, operator.dN_dx)
    mu, lam = nh_lame_parameters(operator)
    derivative, invalid = wang2014_energy_rho_derivative(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_invalid=True,
        **_wang_kwargs(gamma_mode, q, beta0, eta0),
    )
    if bool(torch.any(invalid).item()):
        raise RuntimeError("cannot evaluate Wang2014 sensitivity at invalid F_gamma")
    return torch.sum(derivative * operator.integration_weights[None, :, :], dim=(0, 2))
