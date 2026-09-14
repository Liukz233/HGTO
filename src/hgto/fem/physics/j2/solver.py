"""Incremental small-strain plane-strain J2 equilibrium on Q4 elements.

Newton iterations use a closed-form return mapping. Plastic strain and
hardening history are committed only after convergence of each load
increment. Tangents use the same geometry and integration buffers as
the graph mechanics operator. The optional linear_solver callback takes
explicit element tangents; otherwise the configured iterative solver
is used. Nonconvergence raises a SolveFailure."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hgto.fem.physics.j2.constitutive import (
    j2_return_mapping,
    plane_strain_to_tensor,
    simp_scaled_j2_material,
    tensor_stress_to_plane_voigt,
)
from hgto.fem.kernels import gauss_strains, internal_force
from hgto.fem.solvers.linear import SolveFailure, pcg_callback

_VOIGT_PAIRS = ((0, 0), (1, 1), (0, 1))


def tangent_to_plane_voigt(tangent: torch.Tensor) -> torch.Tensor:
    """Reduce the ``(..., 3, 3, 3, 3)`` tangent to the plane ``(..., 3, 3)``
    engineering-Voigt matrix.

    With engineering shear in the strain vector the (i,j)-(k,l) entry maps
    verbatim (the factor 2 of the tensor shear is absorbed by ``gxy``; the
    tangent's minor symmetry makes this exact).
    """
    rows = [
        torch.stack([tangent[..., i, j, k, l] for k, l in _VOIGT_PAIRS], dim=-1)
        for i, j in _VOIGT_PAIRS
    ]
    return torch.stack(rows, dim=-2)


def _engineering_b_operator(dN_dx: torch.Tensor) -> torch.Tensor:
    """(Ne, 4, 3, 8) engineering-strain B (Jacobi diagonals + element tangents)."""
    B = dN_dx.new_zeros((dN_dx.shape[0], 4, 3, 8))
    B[:, :, 0, 0::2] = dN_dx[..., 0]
    B[:, :, 1, 1::2] = dN_dx[..., 1]
    B[:, :, 2, 0::2] = dN_dx[..., 1]
    B[:, :, 2, 1::2] = dN_dx[..., 0]
    return B


def element_tangent_matrices(
    B: torch.Tensor, D_voigt: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """(Ne, 8, 8) consistent-tangent element matrices ``sum_g w_g B^T D_g B``.

    One-index generalization of the Jacobi-diagonal einsum; exactly the
    matrix of the matrix-free tangent matvec.  ``D_voigt`` is ``(1, Ne, 4, 3, 3)``.
    """
    return torch.einsum("egip,legij,egjq,eg->lepq", B, D_voigt, B, weights)[0]


def mgcg_tangent_preconditioner(operator, element_matrices: torch.Tensor):
    """One V(1,1) Galerkin-telescope preconditioner on the ACTUAL tangent.

    J2 multigrid preconditioning: builds :class:`MGHierarchy` from the
    assembled per-element consistent-tangent matrices (the hierarchy's
    Galerkin coarsening, Gershgorin/Chebyshev smoother bounds and coarsest
    Cholesky are generic over per-element matrices).  Requires the binding
    structured grid — the hierarchy raises otherwise; the caller opted in.
    Returns a reduced-space ``residual -> z`` closure for
    :func:`hgto.fem.solvers.mgcg.preconditioned_cg`.
    """
    from hgto.fem.solvers.mgcg import MGHierarchy

    hierarchy = MGHierarchy(
        operator.Ke0,
        element_matrices.new_ones(operator.n_elements),  # ignored (see below)
        operator.econn,
        operator.free_dof_mask,
        operator.mesh_nelx,
        operator.mesh_nely,
        element_matrices=element_matrices,
    )
    free = operator.free_dof_mask
    n_dof = operator.n_dof

    def precondition(residual_reduced: torch.Tensor) -> torch.Tensor:
        full = residual_reduced.new_zeros(n_dof)
        full[free] = residual_reduced
        return hierarchy.apply(full)[free]

    return precondition


@torch.no_grad()
def solve_j2_history(
    operator,
    rho: torch.Tensor,
    force_steps: torch.Tensor,
    sigma_y0: float,
    H: float,
    rtol: float = 1.0e-10,
    max_newton: int = 50,
    pcg_rtol: float = 1.0e-12,
    max_pcg: int = 20000,
    preconditioner: str = "jacobi",
    linear_solver: Optional[Callable] = None,
) -> dict[str, Any]:
    """Solve a load-stepped history; ``force_steps`` is ``(n_steps, Nn, 2)``.

    ``E0 / Emin / nu / p`` come from the operator (single source of truth);
    ``sigma_y0`` / ``H`` are the FULL-SOLID yield stress and hardening
    modulus — the SIMP triple scaling happens here, never at call sites.

    ``preconditioner`` selects the tangent-solve preconditioner:
    ``"jacobi"`` (default) or ``"mgcg"``
    (optional: Galerkin telescope built on the assembled
    consistent tangent each Newton iteration; requires the binding
    structured grid, same certificates).

    Returns ``history`` (per step: ``u``, ``alpha``, ``plastic_strain``,
    ``residual_rel``, ``newton_iters``, ``pcg_its``, ``plastic_fraction``)
    plus the final internal state. Non-convergence raises a typed
    :class:`SolveFailure` (kind="newton") — failure is data upstream.
    """
    dtype, device = operator.dtype, operator.device
    econn = operator.econn
    dN_dx = operator.dN_dx
    weights = operator.integration_weights
    free = operator.free_dof_mask
    n_nodes, n_elements = operator.n_nodes, operator.n_elements

    forces = torch.as_tensor(force_steps, dtype=dtype, device=device)
    if forces.ndim != 3 or forces.shape[1:] != (n_nodes, 2):
        raise ValueError("force_steps must have shape (n_steps, Nn, 2)")
    if preconditioner not in ("jacobi", "mgcg"):
        raise ValueError("preconditioner must be 'jacobi' or 'mgcg'")
    operator._check_rho(rho)

    E_e, sy_e, H_e = simp_scaled_j2_material(
        rho,
        float(operator.E0),
        float(operator.Emin),
        float(operator.p),
        float(sigma_y0),
        float(H),
    )
    # per-Gauss broadcast fields: (1, Ne, 1) against (L=1, Ne, 4) points
    E_f = E_e.reshape(1, n_elements, 1)
    sy_f = sy_e.reshape(1, n_elements, 1)
    H_f = H_e.reshape(1, n_elements, 1)
    nu = operator.nu

    B = _engineering_b_operator(dN_dx)
    edofs = econn.new_empty((n_elements, 8))
    edofs[:, 0::2] = 2 * econn
    edofs[:, 1::2] = 2 * econn + 1

    plastic = torch.zeros((1, n_elements, 4, 3, 3), dtype=dtype, device=device)
    alpha = torch.zeros((1, n_elements, 4), dtype=dtype, device=device)
    u = torch.zeros((1, n_nodes, 2), dtype=dtype, device=device)

    history: list[dict[str, Any]] = []
    for step in range(forces.shape[0]):
        target = forces[step : step + 1]
        target_reduced = target.reshape(-1)[free]
        target_norm = torch.linalg.vector_norm(target_reduced)
        normalizer = target_norm if float(target_norm.item()) > 0.0 else target_norm.new_tensor(1.0)
        plastic_trial, alpha_trial = plastic, alpha
        residual_rel = float("inf")
        newton_iters = 0
        pcg_total = 0
        converged = False
        for _ in range(max_newton):
            strain = plane_strain_to_tensor(gauss_strains(u, econn, dN_dx))
            stress, plastic_trial, alpha_trial, tangent, plastic_mask = j2_return_mapping(
                strain, plastic, alpha, E_f, nu, sy_f, H_f
            )
            f_int = internal_force(
                tensor_stress_to_plane_voigt(stress),
                econn,
                dN_dx,
                weights,
                n_nodes=n_nodes,
            )
            residual_reduced = (target - f_int).reshape(-1)[free]
            residual_rel = float((torch.linalg.vector_norm(residual_reduced) / normalizer).item())
            if residual_rel <= rtol:
                converged = True
                break
            newton_iters += 1
            D = tangent_to_plane_voigt(tangent)

            def matvec(vector: torch.Tensor) -> torch.Tensor:
                field = vector.new_zeros(operator.n_dof)
                field[free] = vector
                strain_dir = gauss_strains(field.reshape(1, n_nodes, 2), econn, dN_dx)
                stress_dir = torch.einsum("legij,legj->legi", D, strain_dir)
                product = internal_force(stress_dir, econn, dN_dx, weights, n_nodes=n_nodes)
                return product.reshape(-1)[free]

            if linear_solver is not None:
                increment, _, pcg_its, pcg_ok = linear_solver(
                    operator,
                    element_tangent_matrices(B, D, weights),
                    residual_reduced,
                    matvec,
                    pcg_rtol,
                )
            elif preconditioner == "mgcg":
                from hgto.fem.solvers.mgcg import preconditioned_cg

                precondition = mgcg_tangent_preconditioner(
                    operator, element_tangent_matrices(B, D, weights)
                )
                increment, _, pcg_its, pcg_ok = preconditioned_cg(
                    matvec,
                    residual_reduced,
                    precondition,
                    torch.zeros_like(residual_reduced),
                    pcg_rtol,
                    max_pcg,
                )
            else:
                element_diag = torch.einsum("egip,legij,egjp,eg->lep", B, D, B, weights)
                diagonal_full = u.new_zeros(operator.n_dof)
                diagonal_full.index_add_(0, edofs.reshape(-1), element_diag.reshape(-1))
                increment, _, pcg_its, pcg_ok = pcg_callback(
                    matvec,
                    residual_reduced,
                    diagonal_full[free],
                    torch.zeros_like(residual_reduced),
                    pcg_rtol,
                    max_pcg,
                )
            pcg_total += int(pcg_its)
            if not pcg_ok:
                raise SolveFailure(
                    f"J2 Newton PCG stalled at step {step} (newton {newton_iters})",
                    kind="pcg",
                    residual_rel=residual_rel,
                    iterations=int(pcg_its),
                )
            update = u.new_zeros(operator.n_dof)
            update[free] = increment
            u = u + update.reshape(1, n_nodes, 2)
        if not converged:
            raise SolveFailure(
                f"J2 Newton failed at step {step}: residual_rel="
                f"{residual_rel:.6e} after {newton_iters} iterations",
                kind="newton",
                residual_rel=residual_rel,
                iterations=newton_iters,
            )
        plastic, alpha = plastic_trial, alpha_trial
        history.append(
            {
                "u": u.clone(),
                "alpha": alpha.clone(),
                "plastic_strain": plastic.clone(),
                "residual_rel": residual_rel,
                "newton_iters": newton_iters,
                "pcg_its": pcg_total,
                "plastic_fraction": float(plastic_mask.double().mean().item()),
            }
        )
    return {"history": history, "plastic_strain": plastic, "alpha": alpha, "u": u}
