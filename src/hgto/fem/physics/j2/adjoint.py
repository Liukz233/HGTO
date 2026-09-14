"""Transient discrete adjoint for terminal elastoplastic compliance.

A backward recursion accounts for equilibrium and per-Gauss internal
state updates over the full committed loading history. Constitutive
VJPs re-evaluate the closed-form return map once per load increment;
Newton iterations are never unrolled. Equilibrium residual derivatives
carry integration weights; internal-update derivatives do not. The
SIMP scalar scales E, yield stress and hardening together. The optional
linear_solver callback is applied to forward and adjoint tangents."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hgto.fem.physics.j2.constitutive import (
    j2_return_mapping,
    plane_strain_to_tensor,
    tensor_stress_to_plane_voigt,
)
from hgto.fem.physics.j2.solver import (
    _engineering_b_operator,
    element_tangent_matrices,
    mgcg_tangent_preconditioner,
    solve_j2_history,
    tangent_to_plane_voigt,
)
from hgto.fem.kernels import gauss_strains, internal_force
from hgto.fem.solvers.linear import SolveFailure, pcg_callback


def _simp_scale_and_derivative(
    rho: torch.Tensor, E0: float, Emin: float, p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``s(rho) = (Emin + rho^p (E0-Emin)) / E0`` and ``ds/drho``."""
    scale = (Emin + torch.pow(rho, p) * (E0 - Emin)) / E0
    dscale = p * torch.pow(rho, p - 1.0) * (E0 - Emin) / E0
    return scale, dscale


def _tangent_system(operator, D_voigt: torch.Tensor, B: torch.Tensor, edofs: torch.Tensor):
    """(matvec, reduced diagonal) of the reduced consistent tangent."""
    econn, dN_dx = operator.econn, operator.dN_dx
    weights, free = operator.integration_weights, operator.free_dof_mask
    n_nodes = operator.n_nodes

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        field = vector.new_zeros(operator.n_dof)
        field[free] = vector
        strain_dir = gauss_strains(field.reshape(1, n_nodes, 2), econn, dN_dx)
        stress_dir = torch.einsum("legij,legj->legi", D_voigt, strain_dir)
        product = internal_force(stress_dir, econn, dN_dx, weights, n_nodes=n_nodes)
        return product.reshape(-1)[free]

    element_diag = torch.einsum("egip,legij,egjp,eg->lep", B, D_voigt, B, weights)
    diagonal_full = D_voigt.new_zeros(operator.n_dof)
    diagonal_full.index_add_(0, edofs.reshape(-1), element_diag.reshape(-1))
    return matvec, diagonal_full[free]


def transient_compliance_sensitivity(
    operator,
    rho: torch.Tensor,
    force_steps: torch.Tensor,
    sigma_y0: float,
    H: float,
    rtol: float = 1.0e-10,
    pcg_rtol: float = 1.0e-12,
    max_pcg: int = 20000,
    forward: dict[str, Any] | None = None,
    preconditioner: str = "jacobi",
    linear_solver: Optional[Callable] = None,
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    """Return ``(compliance, dC/drho (Ne,), forward_result)``.

    ``force_steps`` is ``(n_steps, Nn, 2)``; the objective is the FINAL
    step's compliance ``f^N . u^N``.  Pass ``forward`` to reuse an existing
    :func:`solve_j2_history` result (it must match the arguments).  A
    single-step history reduces exactly to the classic one-solve adjoint.
    ``preconditioner`` ("jacobi" default | "mgcg", optional)
    selects the tangent-solve preconditioner for BOTH the forward history
    and the adjoint sweeps; certificates are unchanged.
    """
    if preconditioner not in ("jacobi", "mgcg"):
        raise ValueError("preconditioner must be 'jacobi' or 'mgcg'")
    dtype, device = operator.dtype, operator.device
    econn, dN_dx = operator.econn, operator.dN_dx
    weights, free = operator.integration_weights, operator.free_dof_mask
    n_nodes, n_elements = operator.n_nodes, operator.n_elements
    E0 = float(operator.E0)
    Emin = float(operator.Emin)
    p = float(operator.p)
    nu = operator.nu

    forces = torch.as_tensor(force_steps, dtype=dtype, device=device)
    if forward is None:
        forward = solve_j2_history(
            operator,
            rho,
            forces,
            sigma_y0,
            H,
            rtol=rtol,
            pcg_rtol=pcg_rtol,
            max_pcg=max_pcg,
            preconditioner=preconditioner,
            linear_solver=linear_solver,
        )
    history = forward["history"]
    n_steps = len(history)

    zero_plastic = torch.zeros((1, n_elements, 4, 3, 3), dtype=dtype, device=device)
    zero_alpha = torch.zeros((1, n_elements, 4), dtype=dtype, device=device)
    plastic_states = [zero_plastic] + [step["plastic_strain"] for step in history]
    alpha_states = [zero_alpha] + [step["alpha"] for step in history]

    compliance = float(torch.sum(forces[n_steps - 1] * history[-1]["u"][0]).item())
    scale_e, dscale_e = _simp_scale_and_derivative(rho, E0, Emin, p)

    B = _engineering_b_operator(dN_dx)
    edofs = econn.new_empty((n_elements, 8))
    edofs[:, 0::2] = 2 * econn
    edofs[:, 1::2] = 2 * econn + 1
    weight_field = weights[None, :, :, None]

    mu_plastic = zero_plastic.clone()
    mu_alpha = zero_alpha.clone()
    gradient = torch.zeros(n_elements, dtype=dtype, device=device)

    for step in range(n_steps - 1, -1, -1):
        u = history[step]["u"]
        plastic_prev = plastic_states[step]
        alpha_prev = alpha_states[step]
        strain_voigt = gauss_strains(u, econn, dN_dx)

        # consistent tangent at the converged state (no grad needed)
        with torch.no_grad():
            scale_b = scale_e.reshape(1, n_elements, 1)
            _, _, _, tangent, _ = j2_return_mapping(
                plane_strain_to_tensor(strain_voigt),
                plastic_prev,
                alpha_prev,
                scale_b * E0,
                nu,
                scale_b * sigma_y0,
                scale_b * H,
            )
            D_voigt = tangent_to_plane_voigt(tangent)
        matvec, diagonal = _tangent_system(operator, D_voigt, B, edofs)

        # ONE differentiable constitutive re-evaluation feeds every VJP
        strain_leaf = strain_voigt.detach().clone().requires_grad_(True)
        plastic_leaf = plastic_prev.detach().clone().requires_grad_(True)
        alpha_leaf = alpha_prev.detach().clone().requires_grad_(True)
        scale_leaf = scale_e.detach().clone().requires_grad_(True)
        scale_bcast = scale_leaf.reshape(1, n_elements, 1)
        stress_t, plastic_out, alpha_out, _, _ = j2_return_mapping(
            plane_strain_to_tensor(strain_leaf),
            plastic_leaf,
            alpha_leaf,
            scale_bcast * E0,
            nu,
            scale_bcast * sigma_y0,
            scale_bcast * H,
        )
        stress_out = tensor_stress_to_plane_voigt(stress_t)

        # (dG/d[strain, state_prev, scale])^T mu — G carries NO wdetJ
        g_strain, g_plastic, g_alpha, g_scale = torch.autograd.grad(
            [stress_out, plastic_out, alpha_out],
            [strain_leaf, plastic_leaf, alpha_leaf, scale_leaf],
            grad_outputs=[torch.zeros_like(stress_out), mu_plastic, mu_alpha],
            retain_graph=True,
        )
        dJ_du = (
            forces[step : step + 1]
            if step == n_steps - 1
            else torch.zeros((1, n_nodes, 2), dtype=dtype, device=device)
        )
        # (dG/du)^T mu assembles B^T WITHOUT wdetJ: divide the weights back
        # out of internal_force before it multiplies them in again.
        g_nodal = internal_force(g_strain / weight_field, econn, dN_dx, weights, n_nodes=n_nodes)
        rhs_reduced = (-dJ_du + g_nodal).reshape(-1)[free]
        if linear_solver is not None:
            lam_reduced, residual_rel, iterations, converged = linear_solver(
                operator,
                element_tangent_matrices(B, D_voigt, weights),
                rhs_reduced,
                matvec,
                pcg_rtol,
            )
        elif preconditioner == "mgcg":
            from hgto.fem.solvers.mgcg import preconditioned_cg

            precondition = mgcg_tangent_preconditioner(
                operator, element_tangent_matrices(B, D_voigt, weights)
            )
            lam_reduced, residual_rel, iterations, converged = preconditioned_cg(
                matvec,
                rhs_reduced,
                precondition,
                torch.zeros_like(rhs_reduced),
                pcg_rtol,
                max_pcg,
            )
        else:
            lam_reduced, residual_rel, iterations, converged = pcg_callback(
                matvec,
                rhs_reduced,
                diagonal,
                torch.zeros_like(rhs_reduced),
                pcg_rtol,
                max_pcg,
            )
        if not converged:
            raise SolveFailure(
                f"J2 transient adjoint PCG stalled at step {step}",
                kind="pcg",
                residual_rel=float(residual_rel),
                iterations=int(iterations),
            )
        lam_field = u.new_zeros(operator.n_dof)
        lam_field[free] = lam_reduced
        eps_lambda = gauss_strains(lam_field.reshape(1, n_nodes, 2), econn, dN_dx)

        # (dR/d[state_prev, scale])^T lambda — R carries wdetJ via the cotangent
        r_plastic, r_alpha, r_scale = torch.autograd.grad(
            [stress_out, plastic_out, alpha_out],
            [plastic_leaf, alpha_leaf, scale_leaf],
            grad_outputs=[
                eps_lambda * weight_field,
                torch.zeros_like(plastic_out),
                torch.zeros_like(alpha_out),
            ],
            retain_graph=False,
        )
        gradient = gradient + (r_scale - g_scale) * dscale_e
        if step > 0:
            mu_plastic = -r_plastic + g_plastic
            mu_alpha = -r_alpha + g_alpha
    return compliance, gradient, forward


def single_step_compliance_sensitivity(
    operator,
    rho: torch.Tensor,
    force: torch.Tensor,
    sigma_y0: float,
    H: float,
    rtol: float = 1.0e-10,
    pcg_rtol: float = 1.0e-12,
    max_pcg: int = 20000,
    linear_solver=None,
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    """One proportional step from the stress-free state.

    The transient recursion with one load step reduces to the single-step
    adjoint because there are no preceding internal states.
    """
    field = torch.as_tensor(force, dtype=operator.dtype, device=operator.device)
    steps = field.reshape(1, operator.n_nodes, 2)
    return transient_compliance_sensitivity(
        operator,
        rho,
        steps,
        sigma_y0,
        H,
        rtol=rtol,
        pcg_rtol=pcg_rtol,
        max_pcg=max_pcg,
        linear_solver=linear_solver,
    )
