"""Incremental Newton solver for Wang-interpolated 2D Neo-Hookean mechanics.

Every accepted load increment satisfies equilibrium and an admissible
kinematic state. Potential-based Armijo backtracking rejects invalid
Jacobians. The default tangent solver uses PCG with a MINRES fallback;
an optional explicit linear_solver callback receives the exact element
tangent, right-hand side and matrix action. No iteration is retained
in an autograd graph. The default Lame calibration matches linear plane
stress at small strain, not exact finite-strain plane stress."""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import torch

from hgto.fem.physics.neohookean.constitutive import (
    deformation_gradient,
    deformation_jacobian,
    nh_internal_force,
    nh_tangent_diagonal,
    nh_tangent_matvec,
    wang2014_energy_gp,
    wang2014_first_piola,
    wang2014_tangent,
)
from hgto.fem.state import MechanicsState

Matvec = Callable[[torch.Tensor], torch.Tensor]

# Wang-2014 production constants (paper values; overridable per call).
GAMMA_MODE = "heaviside"
GAMMA_Q = 3.0
BETA0 = 500.0
ETA0 = 0.01


def nh_lame_parameters(operator) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D-calibrated Lame pair from the operator's (E0, nu) buffers."""
    E0, nu = operator.E0, operator.nu
    mu = E0 / (2.0 * (1.0 + nu))
    lam = E0 * nu / (1.0 - nu * nu)
    return mu, lam


def _wang_kwargs(gamma_mode: str, gamma_q: float, beta0: float, eta0: float) -> dict:
    return {"gamma_mode": gamma_mode, "q": gamma_q, "beta0": beta0, "eta0": eta0}


def wang2014_internal_force_at(
    operator,
    u: torch.Tensor,
    rho: torch.Tensor,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (f_int, F, invalid) for the Wang-2014 interpolated solid."""
    mu, lam = nh_lame_parameters(operator)
    F = deformation_gradient(operator._masked(u), operator.econn, operator.dN_dx)
    stress, invalid = wang2014_first_piola(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_invalid=True,
        **_wang_kwargs(gamma_mode, q, beta0, eta0),
    )
    force = nh_internal_force(
        stress,
        operator.econn,
        operator.dN_dx,
        operator.integration_weights,
        n_nodes=operator.n_nodes,
    )
    return force, F, invalid


def wang2014_energy_at(
    operator,
    u: torch.Tensor,
    rho: torch.Tensor,
    force: torch.Tensor,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
) -> dict:
    """Return strain energy, external work, total potential, invalid mask."""
    mu, lam = nh_lame_parameters(operator)
    u_free = operator._masked(u)
    F = deformation_gradient(u_free, operator.econn, operator.dN_dx)
    energy_gp, invalid = wang2014_energy_gp(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_invalid=True,
        **_wang_kwargs(gamma_mode, q, beta0, eta0),
    )
    strain_energy = torch.sum(energy_gp * operator.integration_weights[None, :, :])
    external_work = torch.sum(u_free * force)
    return {
        "strain_energy": strain_energy,
        "external_work": external_work,
        "potential": strain_energy - external_work,
        "invalid": invalid,
    }


def wang2014_tangent_at(
    operator,
    u: torch.Tensor,
    rho: torch.Tensor,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (packed tangent, F, invalid) at the given displacement."""
    mu, lam = nh_lame_parameters(operator)
    F = deformation_gradient(operator._masked(u), operator.econn, operator.dN_dx)
    tangent, invalid = wang2014_tangent(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_invalid=True,
        **_wang_kwargs(gamma_mode, q, beta0, eta0),
    )
    return tangent, F, invalid


# --------------------------------------------------------------------------
# reduced-space Krylov callbacks
# --------------------------------------------------------------------------


def pcg_callback(
    matvec: Matvec,
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    initial: torch.Tensor,
    rtol: float,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Deterministic Jacobi-PCG on a reduced-space callback.

    The classic residual recurrence is kept (a freshly assembled residual
    every iteration destroys conjugacy at fp64 roundoff on ill-conditioned
    slender tangents); certificates are re-measured against the TRUE
    residual before returning converged (D-POL-5).
    """
    if rhs.ndim != 1 or diagonal.shape != rhs.shape or initial.shape != rhs.shape:
        raise ValueError("PCG rhs, diagonal, and initial must be equal-sized vectors")
    if rtol <= 0.0 or max_iter <= 0:
        raise ValueError("PCG requires positive rtol and max_iter")
    if bool(torch.any(~torch.isfinite(diagonal)).item()) or bool(torch.any(diagonal <= 0.0).item()):
        # A non-positive Jacobi diagonal already certifies indefiniteness:
        # report non-convergence so the MINRES strategy takes over.
        return initial.clone(), rhs.new_tensor(float("inf")), 0, False

    x = initial.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    normalizer = rhs_norm if float(rhs_norm.item()) > 0.0 else rhs.new_tensor(1.0)
    residual = rhs - matvec(x)
    residual_rel = torch.linalg.vector_norm(residual) / normalizer
    if not bool(torch.isfinite(residual_rel).item()):
        return x, residual_rel, 0, False
    if float(residual_rel.item()) <= rtol:
        return x, residual_rel, 0, True

    z = residual / diagonal
    direction = z.clone()
    rz = torch.dot(residual, z)
    iterations = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        product = matvec(direction)
        curvature = torch.dot(direction, product)
        if not bool(torch.isfinite(curvature).item()) or float(curvature.item()) <= 0.0:
            break  # negative curvature: indefinite tangent, hand to MINRES
        alpha = rz / curvature
        if not bool(torch.isfinite(alpha).item()):
            break
        x = x + alpha * direction
        residual = residual - alpha * product
        residual_rel = torch.linalg.vector_norm(residual) / normalizer
        iterations = iteration
        if not bool(torch.isfinite(residual_rel).item()):
            break
        if float(residual_rel.item()) <= rtol:
            # Certify against the true reduced equation; restart from the
            # true residual if recurrence drift triggered early.
            residual = rhs - matvec(x)
            residual_rel = torch.linalg.vector_norm(residual) / normalizer
            if not bool(torch.isfinite(residual_rel).item()):
                break
            if float(residual_rel.item()) <= rtol:
                converged = True
                break
            z = residual / diagonal
            direction = z.clone()
            rz = torch.dot(residual, z)
            if not bool(torch.isfinite(rz).item()) or float(rz.item()) == 0.0:
                break
            continue
        z = residual / diagonal
        rz_new = torch.dot(residual, z)
        if not bool(torch.isfinite(rz_new).item()) or float(rz.item()) == 0.0:
            break
        beta = rz_new / rz
        if not bool(torch.isfinite(beta).item()):
            break
        direction = z + beta * direction
        rz = rz_new
    if not converged:
        residual = rhs - matvec(x)
        residual_rel = torch.linalg.vector_norm(residual) / normalizer
    return x, residual_rel, iterations, converged


def minres_callback(
    matvec: Matvec,
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    initial: torch.Tensor,
    rtol: float,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Matrix-free symmetric-INDEFINITE solve (the PCG breakdown remedy).

    SciPy MINRES over the same torch matvec with an
    absolute-value Jacobi SPD preconditioner. SciPy's internal tolerance is
    a conservative estimate, so it is driven to the floor and the returned
    solution is certified against ``rtol`` with the TRUE torch residual
    (D-POL-5). Drop-in signature-compatible with :func:`pcg_callback`.
    """
    if rhs.ndim != 1 or diagonal.shape != rhs.shape or initial.shape != rhs.shape:
        raise ValueError("MINRES rhs, diagonal, and initial must be equal-sized vectors")
    if rtol <= 0.0 or max_iter <= 0:
        raise ValueError("MINRES requires positive rtol and max_iter")

    import numpy as np
    from scipy.sparse.linalg import LinearOperator, minres

    dtype, device = rhs.dtype, rhs.device
    n = int(rhs.shape[0])

    def _matvec_np(vector: "np.ndarray") -> "np.ndarray":
        field = torch.as_tensor(np.ascontiguousarray(vector), dtype=dtype, device=device)
        return matvec(field).detach().cpu().numpy().astype(np.float64)

    linear_operator = LinearOperator((n, n), matvec=_matvec_np, dtype=np.float64)
    diag_abs = np.maximum(
        np.abs(diagonal.detach().cpu().numpy().astype(np.float64)),
        np.finfo(np.float64).tiny,
    )
    preconditioner = LinearOperator(
        (n, n), matvec=lambda vector: vector / diag_abs, dtype=np.float64
    )
    counter = {"iterations": 0}

    def _count(_x: "np.ndarray") -> None:
        counter["iterations"] += 1

    rhs_np = rhs.detach().cpu().numpy().astype(np.float64)
    rhs_norm = torch.linalg.vector_norm(rhs)
    normalizer = rhs_norm if float(rhs_norm.item()) > 0.0 else rhs.new_tensor(1.0)

    # SciPy MINRES stops early on its internal ill-conditioning estimates
    # (observed: istop exit at ~500 its with residual 1e-6-class on
    # gray-field NH tangents). RESTART from the returned iterate until the
    # true residual stalls or meets rtol — each restart resets the Lanczos
    # basis, which is the standard remedy.
    x_np = initial.detach().cpu().numpy().astype(np.float64)
    best_residual = float("inf")
    for _restart in range(8):
        solution, _info = minres(
            linear_operator,
            rhs_np,
            x0=x_np,
            rtol=min(float(rtol) * 1.0e-4, 1.0e-13),
            maxiter=int(max_iter),
            M=preconditioner,
            callback=_count,
        )
        x_np = np.ascontiguousarray(solution)
        x = torch.as_tensor(x_np, dtype=dtype, device=device)
        residual_rel = torch.linalg.vector_norm(rhs - matvec(x)) / normalizer
        current = float(residual_rel.item())
        if not np.isfinite(current):
            break
        if current <= rtol or current > 0.9 * best_residual:
            best_residual = min(best_residual, current)
            break
        best_residual = current
    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    residual_rel = torch.linalg.vector_norm(rhs - matvec(x)) / normalizer
    converged = bool(torch.isfinite(residual_rel).item()) and (float(residual_rel.item()) <= rtol)
    return x, residual_rel, int(counter["iterations"]), converged


def solve_reduced_tangent(
    matvec: Matvec,
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    rtol_pcg: float,
    rtol_outer: float,
    max_iter: int,
) -> Tuple[torch.Tensor, int, int, bool]:
    """PCG -> MINRES strategy for one reduced tangent system.

    Returns (solution, inner_iterations, fallbacks_used, converged). MINRES
    acceptance is at ``max(rtol_outer, 1e-9)`` — its conditioning floor on FE
    tangents is ~1e-10, and the inexact-Newton bound only needs the outer
    tolerance.
    """
    zero = torch.zeros_like(rhs)
    solution, _, pcg_iterations, converged = pcg_callback(
        matvec, rhs, diagonal, zero, rtol_pcg, max_iter
    )
    if converged:
        return solution, pcg_iterations, 0, True
    fallback_rtol = max(rtol_outer, 1.0e-9)
    solution, _, minres_iterations, converged = minres_callback(
        matvec, rhs, diagonal, zero, fallback_rtol, max_iter
    )
    return solution, pcg_iterations + minres_iterations, 1, converged


# --------------------------------------------------------------------------
# line search + Newton driver
# --------------------------------------------------------------------------


def _line_search(
    operator,
    rho: torch.Tensor,
    force: torch.Tensor,
    current: torch.Tensor,
    direction: torch.Tensor,
    residual_reduced: torch.Tensor,
    max_backtracks: int,
    armijo: float,
    wang: dict,
) -> Tuple[torch.Tensor, int, bool]:
    """NaN-aware Armijo on total potential; NEVER accepts exhaustion.

    Globalization (nonconvex regime): if the (inexact) Newton direction is
    not a descent direction — possible on indefinite tangents, where
    ``-K^{-1} r`` may point uphill — swap to the NEGATIVE-GRADIENT
    direction, rescaled to the Newton direction's magnitude. It is descent
    by construction, Armijo then guarantees decrease, and the swap is
    reported through the third return value (never silent). A NON-FINITE
    slope still raises: that is numerical pathology, not geometry.
    """
    current_energy = wang2014_energy_at(operator, current, rho, force, **wang)
    current_potential = current_energy["potential"]
    if bool(torch.any(current_energy["invalid"]).item()) or not bool(
        torch.isfinite(current_potential).item()
    ):
        raise RuntimeError("NH Newton iterate has invalid total potential")
    direction_reduced = direction.reshape(-1)[operator.free_dof_mask]
    slope = torch.dot(residual_reduced, direction_reduced)
    if not bool(torch.isfinite(slope).item()):
        raise RuntimeError("NH Newton direction slope is not finite")
    used_gradient = False
    if float(slope.item()) >= 0.0:
        used_gradient = True
        newton_norm = torch.linalg.vector_norm(direction_reduced)
        gradient_norm = torch.linalg.vector_norm(residual_reduced)
        if float(gradient_norm.item()) == 0.0 or not bool(torch.isfinite(newton_norm).item()):
            raise RuntimeError("NH gradient fallback direction is degenerate")
        scale = newton_norm / gradient_norm
        if not bool(torch.isfinite(scale).item()) or float(scale.item()) == 0.0:
            scale = residual_reduced.new_tensor(1.0)
        direction = torch.zeros_like(direction)
        direction.reshape(-1)[operator.free_dof_mask] = -scale * residual_reduced
        direction_reduced = direction.reshape(-1)[operator.free_dof_mask]
        slope = torch.dot(residual_reduced, direction_reduced)  # = -scale|r|^2 < 0

    step = 1.0
    for backtracks in range(max_backtracks + 1):
        candidate = operator._masked(current + step * direction)
        energy = wang2014_energy_at(operator, candidate, rho, force, **wang)
        potential = energy["potential"]
        valid = (
            not bool(torch.any(energy["invalid"]).item())
            and bool(torch.all(torch.isfinite(candidate)).item())
            and bool(torch.isfinite(potential).item())
        )
        if valid:
            bound = current_potential + armijo * step * slope
            # nan > gate is silently False — the finite check above stays
            # explicit; allow fp roundoff headroom.
            roundoff = (
                torch.finfo(potential.dtype).eps * 32.0 * (1.0 + torch.abs(current_potential))
            )
            if float(potential.item()) <= float((bound + roundoff).item()):
                return candidate, backtracks, used_gradient
        step *= 0.5
    raise RuntimeError(
        f"NH energy line search exhausted {max_backtracks} backtracks; step rejected"
    )


@torch.no_grad()
def solve_nh_state(
    operator,
    rho: torch.Tensor,
    f_ext: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    n_ramp: int = 5,
    rtol: float = 1.0e-8,
    max_newton: int = 40,
    max_inner: int = 20000,
    pcg_rtol: float = 1.0e-11,
    max_backtracks: int = 24,
    armijo: float = 1.0e-4,
    warm_start: bool = True,
    gamma_mode: str = GAMMA_MODE,
    q: float = GAMMA_Q,
    beta0: float = BETA0,
    eta0: float = ETA0,
    record_ramp_history: bool = False,
    linear_solver: Optional[Callable] = None,
) -> MechanicsState:
    """Solve every load case of the Wang-2014 NH solid.

    Residual is ``f_int - lambda f_ext`` over a monotone load ramp; each
    tangent system runs the PCG->MINRES strategy; each accepted step passes
    the potential-based Armijo search. Failure is loud (typed RuntimeError),
    never a silent NaN state.

    ``record_ramp_history=True`` additionally CLONES the converged
    displacement of every accepted ramp step into ``state.ramp_u`` (shape
    ``(L, n_ramp, n_nodes, 2)`` — the X2 F-d exhibit source). Recording is
    purely observational: the solve path and every returned certificate are
    bitwise identical with the flag on or off (default off).
    """
    started = time.perf_counter()
    operator._check_rho(rho)
    operator._check_vector_field(f_ext, "f_ext")
    if n_ramp <= 0 or max_newton <= 0 or max_inner <= 0:
        raise ValueError("n_ramp, max_newton, and max_inner must be positive")
    if rtol <= 0.0 or pcg_rtol <= 0.0:
        raise ValueError("Newton and inner tolerances must be positive")
    if max_backtracks < 0 or not 0.0 < armijo < 1.0:
        raise ValueError("invalid line-search controls")
    wang = _wang_kwargs(gamma_mode, q, beta0, eta0)
    if u0 is None:
        seed = torch.zeros_like(f_ext)
    else:
        operator._check_vector_field(u0, "u0")
        if u0.shape != f_ext.shape:
            raise ValueError("u0 and f_ext must have identical shapes")
        seed = operator._masked(u0)

    load_factors = torch.arange(
        1, n_ramp + 1, dtype=operator.dtype, device=operator.device
    ) / float(n_ramp)
    final_u, final_residual = [], []
    newton_totals, inner_totals, backtrack_totals, fallback_totals = [], [], [], []
    inexact_totals = []
    ramp_newton_all, ramp_residual_all = [], []
    ramp_u_all: list[torch.Tensor] = []

    for load_index in range(f_ext.shape[0]):
        previous = torch.zeros_like(f_ext[load_index : load_index + 1])
        seed_one = seed[load_index : load_index + 1]
        force_full = f_ext[load_index : load_index + 1]
        load_newton = load_inner = load_backtracks = load_fallbacks = 0
        load_inexact = 0
        ramp_newton, ramp_residual = [], []
        ramp_u: list[torch.Tensor] = []

        for ramp_index, load_factor in enumerate(load_factors):
            if warm_start and ramp_index > 0:
                current = previous.clone()
            elif u0 is not None:
                current = load_factor * seed_one
            else:
                current = torch.zeros_like(previous)
            target_force = load_factor * force_full
            rhs_reduced = target_force.reshape(-1)[operator.free_dof_mask]
            rhs_norm = torch.linalg.vector_norm(rhs_reduced)
            normalizer = rhs_norm if float(rhs_norm.item()) > 0.0 else rhs_norm.new_tensor(1.0)
            converged = False
            accepted = 0
            residual_rel = rhs_norm.new_tensor(float("inf"))

            for _ in range(max_newton + 1):
                internal, _, invalid = wang2014_internal_force_at(operator, current, rho, **wang)
                if bool(torch.any(invalid).item()) or not bool(
                    torch.all(torch.isfinite(internal)).item()
                ):
                    raise RuntimeError("NH Newton encountered invalid det(F)")
                residual_field = operator._masked(internal - target_force)
                residual_reduced = residual_field.reshape(-1)[operator.free_dof_mask]
                residual_rel = torch.linalg.vector_norm(residual_reduced) / normalizer
                if not bool(torch.isfinite(residual_rel).item()):
                    raise RuntimeError("NH Newton residual is non-finite")
                if float(residual_rel.item()) <= rtol:
                    converged = True
                    break
                if accepted >= max_newton:
                    break

                tangent, _, tangent_invalid = wang2014_tangent_at(operator, current, rho, **wang)
                if bool(torch.any(tangent_invalid).item()) or not bool(
                    torch.all(torch.isfinite(tangent)).item()
                ):
                    raise RuntimeError("NH Newton tangent is invalid")

                def matvec(x: torch.Tensor) -> torch.Tensor:
                    full = x.new_zeros(operator.n_dof)
                    full[operator.free_dof_mask] = x
                    product = nh_tangent_matvec(
                        tangent,
                        full.reshape(1, operator.n_nodes, 2),
                        operator.econn,
                        operator.dN_dx,
                        operator.integration_weights,
                        n_nodes=operator.n_nodes,
                    )
                    return product.reshape(-1)[operator.free_dof_mask]

                if linear_solver is None:
                    diagonal = nh_tangent_diagonal(
                        tangent,
                        operator.econn,
                        operator.dN_dx,
                        operator.integration_weights,
                        operator.n_nodes,
                    )[0, operator.free_dof_mask]
                    increment, inner_iterations, fallbacks, inner_converged = solve_reduced_tangent(
                        matvec, -residual_reduced, diagonal, pcg_rtol, rtol, max_inner
                    )
                else:
                    element_matrices = torch.einsum(
                        "egiJkL,egaJ,egbL,eg->eaibk",
                        tangent[0],
                        operator.dN_dx,
                        operator.dN_dx,
                        operator.integration_weights,
                    ).reshape(operator.n_elements, 8, 8)
                    increment, _, inner_iterations, inner_converged = linear_solver(
                        operator,
                        element_matrices,
                        -residual_reduced,
                        matvec,
                        pcg_rtol,
                    )
                    fallbacks = 0
                load_inner += inner_iterations
                load_fallbacks += fallbacks
                usable = bool(torch.all(torch.isfinite(increment)).item()) and (
                    float(torch.linalg.vector_norm(increment).item()) > 0.0
                )
                if not inner_converged and not usable:
                    # Keep the "PCG failed" phrasing: downstream BW rescue
                    # logic recognizes a recoverable inner-solve failure.
                    raise RuntimeError(
                        "NH Newton PCG failed at ramp {}/{} after {} iterations"
                        " (MINRES strategy also failed)".format(
                            ramp_index + 1, n_ramp, inner_iterations
                        )
                    )
                # Inexact-Newton acceptance: on strongly indefinite gray-field
                # tangents MINRES can stall above the inner target (SciPy
                # istop, ~1e-6-class true residual). A finite stalled
                # direction is still a CANDIDATE — the descent pre-check and
                # the Armijo search below are the quality gates, and the
                # outer certificate (residual <= rtol at convergence) is
                # untouched. Stalls are counted, never silent.
                if not inner_converged:
                    load_inexact += 1
                direction = torch.zeros_like(current)
                direction.reshape(-1)[operator.free_dof_mask] = increment
                current, backtracks, used_gradient = _line_search(
                    operator,
                    rho,
                    target_force,
                    current,
                    direction,
                    residual_reduced,
                    max_backtracks,
                    armijo,
                    wang,
                )
                if used_gradient:
                    load_inexact += 1
                load_backtracks += backtracks
                accepted += 1
                load_newton += 1

            if not converged:
                raise RuntimeError(
                    "NH Newton failed at ramp {}/{}: residual={:.6e}, steps={}".format(
                        ramp_index + 1, n_ramp, float(residual_rel.item()), accepted
                    )
                )
            previous = current
            ramp_newton.append(accepted)
            ramp_residual.append(residual_rel)
            if record_ramp_history:
                # Observational clone of the accepted per-ramp displacement
                # (X2 F-d exhibits); never touches the solve trajectory.
                ramp_u.append(previous[0].clone())

        final_u.append(previous[0])
        final_residual.append(ramp_residual[-1])
        newton_totals.append(load_newton)
        inner_totals.append(load_inner)
        backtrack_totals.append(load_backtracks)
        fallback_totals.append(load_fallbacks)
        inexact_totals.append(load_inexact)
        ramp_newton_all.append(ramp_newton)
        ramp_residual_all.append(torch.stack(ramp_residual))
        if record_ramp_history:
            ramp_u_all.append(torch.stack(ramp_u))

    displacement = torch.stack(final_u, dim=0)
    final_internal, F, invalid = wang2014_internal_force_at(operator, displacement, rho, **wang)
    if bool(torch.any(invalid).item()):
        raise RuntimeError("converged NH state contains invalid det(F)")
    determinant, _ = deformation_jacobian(F)
    mu, lam = nh_lame_parameters(operator)
    physical_energy_gp, energy_bracket_gp = wang2014_energy_gp(
        F,
        rho,
        mu,
        lam,
        operator.p,
        operator.Emin / operator.E0,
        return_bracket=True,
        **wang,
    )
    stress_scaled = wang2014_first_piola(
        F, rho, mu, lam, operator.p, operator.Emin / operator.E0, **wang
    )
    right_cauchy_green = torch.einsum("...kI,...kJ->...IJ", F, F)
    identity = torch.eye(2, dtype=operator.dtype, device=operator.device)
    green = 0.5 * (right_cauchy_green - identity)
    green_voigt = torch.stack([green[..., 0, 0], green[..., 1, 1], 2.0 * green[..., 0, 1]], dim=-1)
    strain_energy = torch.sum(physical_energy_gp * operator.integration_weights[None, :, :])
    compliance = torch.sum(displacement * f_ext)
    elapsed = time.perf_counter() - started
    return MechanicsState(
        u=displacement,
        strain_gp=green_voigt,
        stress_gp=stress_scaled,
        unit_strain_energy_gp=energy_bracket_gp,
        strain_energy_gp=physical_energy_gp,
        f_int=final_internal,
        residual_rel=torch.stack(final_residual),
        iterations=torch.tensor(inner_totals, dtype=torch.long, device=operator.device),
        converged=torch.ones(f_ext.shape[0], dtype=torch.bool, device=operator.device),
        compliance=compliance,
        F_gp=F,
        detF_min=torch.amin(determinant, dim=(1, 2)),
        newton_iterations=torch.tensor(newton_totals, dtype=torch.long, device=operator.device),
        backtracks=torch.tensor(backtrack_totals, dtype=torch.long, device=operator.device),
        pcg_iterations=torch.tensor(inner_totals, dtype=torch.long, device=operator.device),
        load_factors=load_factors,
        indefinite_fallbacks=torch.tensor(
            fallback_totals, dtype=torch.long, device=operator.device
        ),
        inexact_directions=torch.tensor(inexact_totals, dtype=torch.long, device=operator.device),
        strain_energy=strain_energy,
        potential=strain_energy - compliance,
        solver_rtol=torch.full(
            (f_ext.shape[0],), rtol, dtype=operator.dtype, device=operator.device
        ),
        ramp_newton_iterations=torch.tensor(
            ramp_newton_all, dtype=torch.long, device=operator.device
        ),
        ramp_residual_rel=torch.stack(ramp_residual_all),
        ramp_u=torch.stack(ramp_u_all) if record_ramp_history else None,
        t_state_total=elapsed,
    )
