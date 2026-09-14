"""Packed total-Lagrangian Q4 kernels for 2D compressible Neo-Hookean solids.

The two-dimensional energy is zero at the identity deformation. Callers
supply the Lame pair; the default solver uses mu=E/[2(1+nu)] and
lambda=E*nu/(1-nu^2), matching linear plane stress at small strain.
This calibration is not a finite-strain plane-stress reduction.
Explicit 2x2 cofactor formulas evaluate inverse-transposes and log-J
terms. Inverted deformations are identified by an invalid mask, which
callers must reject; stress/tangent invalid entries remain finite while
invalid energy is +inf.

The Wang energy interpolation is
    Psi_int = s(rho) [Psi_NH(I+gamma*grad(u))
                     -Psi_L(gamma*grad(u))+Psi_L(grad(u))].
Gamma interpolates the kinematics and s(rho) is the SIMP stiffness scale
including the prescribed floor. Full density recovers the solid NH law;
gamma=0 recovers linear elasticity. Reference: Wang, Lazarov, Sigmund
and Jensen, CMAME 276 (2014), 453-472, doi:10.1016/j.cma.2014.03.021."""

from __future__ import annotations

from typing import Tuple, Union

import torch

Scalar = Union[float, torch.Tensor]


def _as_scalar(value: Scalar, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _require_2x2(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim < 2 or tensor.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing shape (2, 2)")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")


# --------------------------------------------------------------------------
# cofactor primitives
# --------------------------------------------------------------------------


def deformation_jacobian(F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return hand-written ``det F`` and the pointwise invalid mask."""
    _require_2x2(F, "F")
    det = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
    invalid = (~torch.isfinite(det)) | (det <= 0.0)
    return det, invalid


def inverse_transpose_2x2(
    F: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``F^{-T}``, ``det F``, and the invalid mask, via cofactors.

    Invalid points get ``F^{-T} = 0`` (finite-neutral); the mask is the only
    truth about validity.
    """
    det, invalid = deformation_jacobian(F)
    safe_det = torch.where(invalid, torch.ones_like(det), det)
    row0 = torch.stack([F[..., 1, 1], -F[..., 1, 0]], dim=-1)
    row1 = torch.stack([-F[..., 0, 1], F[..., 0, 0]], dim=-1)
    cofactor = torch.stack([row0, row1], dim=-2)
    inv_t = cofactor / safe_det[..., None, None]
    inv_t = torch.where(invalid[..., None, None], torch.zeros_like(inv_t), inv_t)
    return inv_t, det, invalid


def deformation_gradient(
    u: torch.Tensor,
    econn: torch.Tensor,
    dN_dX: torch.Tensor,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``F = I + grad_X(u)`` at every Gauss point, shape (L, Ne, 4, 2, 2)."""
    if u.ndim != 3 or u.shape[-1] != 2:
        raise ValueError("u must have shape (L, Nn, 2)")
    if econn.ndim != 2 or econn.shape[1] != 4 or econn.dtype != torch.long:
        raise ValueError("econn must have shape (Ne, 4) with dtype torch.long")
    if dN_dX.shape != (econn.shape[0], 4, 4, 2):
        raise ValueError("dN_dX must have shape (Ne, 4, 4, 2)")
    if u.dtype != dN_dX.dtype:
        raise TypeError("u and dN_dX must share a dtype")
    element_u = u[:, econn, :]
    grad_u = torch.einsum("leai,egaJ->legiJ", element_u, dN_dX)
    identity = torch.eye(2, dtype=u.dtype, device=u.device)
    F = grad_u + identity.reshape(1, 1, 1, 2, 2)
    if return_invalid:
        return F, deformation_jacobian(F)[1]
    return F


# --------------------------------------------------------------------------
# plain compressible Neo-Hookean (full solid)
# --------------------------------------------------------------------------


def nh_energy_gp(
    F: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Unscaled NH energy density per Gauss point.

    ``Psi = mu/2 (I_C - 2 - 2 ln J) + lam/2 (ln J)^2`` with
    ``I_C = F:F`` (2D). Invalid points carry ``+inf`` — never NaN.
    """
    _, det, invalid = inverse_transpose_2x2(F)
    safe_det = torch.where(invalid, torch.ones_like(det), det)
    log_j = torch.log(safe_det)
    trace_c = torch.sum(F * F, dim=(-2, -1))
    mu_t, lam_t = _as_scalar(mu, F), _as_scalar(lam, F)
    energy = 0.5 * mu_t * (trace_c - 2.0 - 2.0 * log_j) + 0.5 * lam_t * log_j * log_j
    energy = torch.where(invalid, torch.full_like(energy, float("inf")), energy)
    if return_invalid:
        return energy, invalid
    return energy


def nh_first_piola(
    F: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """First Piola stress ``P = mu (F - F^{-T}) + lam ln(J) F^{-T}``."""
    inv_t, det, invalid = inverse_transpose_2x2(F)
    safe_det = torch.where(invalid, torch.ones_like(det), det)
    log_j = torch.log(safe_det)
    mu_t, lam_t = _as_scalar(mu, F), _as_scalar(lam, F)
    P = mu_t * (F - inv_t) + (lam_t * log_j)[..., None, None] * inv_t
    P = torch.where(invalid[..., None, None], torch.zeros_like(P), P)
    if return_invalid:
        return P, invalid
    return P


def nh_tangent(
    F: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Closed-form packed tangent ``A_{iJkL} = dP_iJ / dF_kL``.

    ``A = mu (delta_ik delta_JL) + lam (F^{-T} outer F^{-T})
    + (mu - lam ln J) (F^{-T}_{iL} F^{-T}_{kJ})`` — derived from
    ``d(F^{-T})/dF`` in cofactor form; no autograd anywhere.
    """
    inv_t, det, invalid = inverse_transpose_2x2(F)
    safe_det = torch.where(invalid, torch.ones_like(det), det)
    log_j = torch.log(safe_det)
    mu_t, lam_t = _as_scalar(mu, F), _as_scalar(lam, F)
    identity = torch.eye(2, dtype=F.dtype, device=F.device)
    kron = torch.einsum("ik,JL->iJkL", identity, identity)
    outer = torch.einsum("...iJ,...kL->...iJkL", inv_t, inv_t)
    crossed = torch.einsum("...iL,...kJ->...iJkL", inv_t, inv_t)
    A = mu_t * kron + lam_t * outer + (mu_t - lam_t * log_j)[..., None, None, None, None] * crossed
    A = torch.where(invalid[..., None, None, None, None], torch.zeros_like(A), A)
    if return_invalid:
        return A, invalid
    return A


# --------------------------------------------------------------------------
# assembly: internal force / tangent action / tangent diagonal
# --------------------------------------------------------------------------


def nh_internal_force(
    P_scaled: torch.Tensor,
    econn: torch.Tensor,
    dN_dX: torch.Tensor,
    wdetJ0: torch.Tensor,
    n_nodes: int | None = None,
) -> torch.Tensor:
    """Integrate and scatter total-Lagrangian nodal internal forces."""
    if P_scaled.ndim != 5 or P_scaled.shape[-3:] != (4, 2, 2):
        raise ValueError("P_scaled must have shape (L, Ne, 4, 2, 2)")
    n_elements = econn.shape[0]
    if P_scaled.shape[1] != n_elements:
        raise ValueError("P_scaled and econn element counts differ")
    if wdetJ0.shape != (n_elements, 4):
        raise ValueError("wdetJ0 must have shape (Ne, 4)")
    element_force = torch.einsum("legiJ,egaJ,eg->leai", P_scaled, dN_dX, wdetJ0)
    if n_nodes is None:
        n_nodes = int(torch.max(econn).item()) + 1
    force = P_scaled.new_zeros((P_scaled.shape[0], n_nodes, 2))
    force.index_add_(1, econn.reshape(-1), element_force.reshape(P_scaled.shape[0], -1, 2))
    return force


def nh_tangent_matvec(
    tangent_scaled: torch.Tensor,
    value: torch.Tensor,
    econn: torch.Tensor,
    dN_dX: torch.Tensor,
    wdetJ0: torch.Tensor,
    n_nodes: int | None = None,
    transpose: bool = False,
) -> torch.Tensor:
    """Assemble the (optionally transposed) action of a packed tangent.

    ``transpose=True`` contracts ``A_{kLiJ}`` instead of ``A_{iJkL}`` —
    the adjoint solves use it so no transposed tensor is ever materialized.
    """
    if tangent_scaled.ndim != 7 or tangent_scaled.shape[-5:] != (4, 2, 2, 2, 2):
        raise ValueError("tangent_scaled must have shape (L, Ne, 4, 2, 2, 2, 2)")
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError("value must have shape (L, Nn, 2)")
    if tangent_scaled.shape[:2] != (value.shape[0], econn.shape[0]):
        raise ValueError("tangent and value load/element counts differ")
    element_value = value[:, econn, :]
    gradient = torch.einsum("leak,egaL->legkL", element_value, dN_dX)
    if transpose:
        delta_P = torch.einsum("legkLiJ,legkL->legiJ", tangent_scaled, gradient)
    else:
        delta_P = torch.einsum("legiJkL,legkL->legiJ", tangent_scaled, gradient)
    return nh_internal_force(delta_P, econn, dN_dX, wdetJ0, n_nodes=n_nodes)


def nh_tangent_diagonal(
    tangent_scaled: torch.Tensor,
    econn: torch.Tensor,
    dN_dX: torch.Tensor,
    wdetJ0: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Full-DOF Jacobi diagonal of every load's packed tangent.

    ``K_(ai,ai) = int N_{a,J} A_{iJ iL} N_{a,L}`` — assembled directly,
    never via dense K.
    """
    if tangent_scaled.ndim != 7 or tangent_scaled.shape[-5:] != (4, 2, 2, 2, 2):
        raise ValueError("tangent_scaled must have shape (L, Ne, 4, 2, 2, 2, 2)")
    element_diag = torch.einsum("legiJiL,egaJ,egaL,eg->leai", tangent_scaled, dN_dX, dN_dX, wdetJ0)
    diagonal = tangent_scaled.new_zeros((tangent_scaled.shape[0], 2 * n_nodes))
    local_dofs = 2 * econn[:, :, None] + torch.arange(
        2, dtype=torch.long, device=econn.device
    ).reshape(1, 1, 2)
    diagonal.index_add_(
        1, local_dofs.reshape(-1), element_diag.reshape(tangent_scaled.shape[0], -1)
    )
    return diagonal


# --------------------------------------------------------------------------
# small-strain (plane-strain) companions — Wang-2014 building blocks
# --------------------------------------------------------------------------


def linear_energy_gp(displacement_gradient: torch.Tensor, mu: Scalar, lam: Scalar) -> torch.Tensor:
    """Plane-strain small-strain energy for ``H = grad_X(u)``.

    Same Lame constants as the ``F33 = 1`` NH restriction (eps_33 = 0):
    ``Psi_L = mu eps:eps + lam/2 tr(eps)^2``.
    """
    _require_2x2(displacement_gradient, "displacement_gradient")
    strain = 0.5 * (displacement_gradient + displacement_gradient.transpose(-2, -1))
    trace = strain[..., 0, 0] + strain[..., 1, 1]
    mu_t = _as_scalar(mu, displacement_gradient)
    lam_t = _as_scalar(lam, displacement_gradient)
    return mu_t * torch.sum(strain * strain, dim=(-2, -1)) + 0.5 * lam_t * trace * trace


def linear_first_piola(
    displacement_gradient: torch.Tensor, mu: Scalar, lam: Scalar
) -> torch.Tensor:
    """``d Psi_L / d H`` for plane-strain linear elasticity."""
    _require_2x2(displacement_gradient, "displacement_gradient")
    trace = displacement_gradient[..., 0, 0] + displacement_gradient[..., 1, 1]
    identity = torch.eye(2, dtype=displacement_gradient.dtype, device=displacement_gradient.device)
    mu_t = _as_scalar(mu, displacement_gradient)
    lam_t = _as_scalar(lam, displacement_gradient)
    return (
        mu_t * (displacement_gradient + displacement_gradient.transpose(-2, -1))
        + (lam_t * trace)[..., None, None] * identity
    )


def linear_tangent(reference: torch.Tensor, mu: Scalar, lam: Scalar) -> torch.Tensor:
    """Constant packed ``d P_L / d H`` (no batch axes — it is H-independent)."""
    _require_2x2(reference, "reference")
    identity = torch.eye(2, dtype=reference.dtype, device=reference.device)
    mu_t = _as_scalar(mu, reference)
    lam_t = _as_scalar(lam, reference)
    direct = torch.einsum("ik,JL->iJkL", identity, identity)
    swapped = torch.einsum("iL,Jk->iJkL", identity, identity)
    volumetric = torch.einsum("iJ,kL->iJkL", identity, identity)
    return mu_t * (direct + swapped) + lam_t * volumetric


# --------------------------------------------------------------------------
# Wang-2014 interpolation family
# --------------------------------------------------------------------------


def wang2014_gamma(
    rho: torch.Tensor,
    mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_derivative: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Kinematic interpolation ``gamma(rho)`` and optionally its slope.

    ``heaviside`` retains the legacy direct-density switch. The explicit
    ``simp_heaviside`` mode applies the switch to ``rho**q``; callers pass
    the current SIMP penalty as q. ``power`` returns ``rho**q`` directly.
    Input is the already filtered/projected physical density in [0, 1].
    """
    if not rho.is_floating_point():
        raise TypeError("rho must be floating point")
    if bool(torch.any(~torch.isfinite(rho)).item()) or bool(
        torch.any((rho < 0.0) | (rho > 1.0)).item()
    ):
        raise ValueError("rho must be finite and lie in [0, 1]")
    mode_name = str(mode).lower()
    if mode_name == "power":
        if not q > 0.0:
            raise ValueError("power-mode exponent q must be positive")
        gamma = torch.pow(rho, q)
        derivative = q * torch.pow(rho, q - 1.0)
    elif mode_name in ("heaviside", "simp_heaviside"):
        if not beta0 > 0.0 or not 0.0 < eta0 < 1.0:
            raise ValueError("heaviside mode requires beta0 > 0 and 0 < eta0 < 1")
        beta = rho.new_tensor(beta0)
        eta = rho.new_tensor(eta0)
        low = torch.tanh(beta * eta)
        if mode_name == "simp_heaviside":
            exponent = torch.as_tensor(q, dtype=rho.dtype, device=rho.device)
            if not bool(exponent >= 1):
                raise ValueError("SIMP exponent must be at least one")
            argument = rho.pow(exponent)
            argument_derivative = exponent * rho.pow(exponent - 1)
        else:
            argument = rho
            argument_derivative = torch.ones_like(rho)
        shifted = torch.tanh(beta * (argument - eta))
        denom = low + torch.tanh(beta * (1.0 - eta))
        gamma = (low + shifted) / denom
        derivative = beta * (1.0 - shifted * shifted) / denom * argument_derivative
    else:
        raise ValueError("wang2014 gamma mode must be 'heaviside', 'simp_heaviside' or 'power'")
    if return_derivative:
        return gamma, derivative
    return gamma


def _wang2014_fields(
    F: torch.Tensor,
    rho: torch.Tensor,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str,
    q: float,
    beta0: float,
    eta0: float,
) -> Tuple[torch.Tensor, ...]:
    """Shared kinematic/scale fields: H, F_gamma, gamma, gamma', s, s'."""
    _require_2x2(F, "F")
    if rho.ndim != 1 or F.ndim < 4 or F.shape[-4] != rho.numel():
        raise ValueError("rho must match the element axis of F")
    if rho.dtype != F.dtype or rho.device != F.device:
        raise TypeError("rho and F must share dtype and device")
    p_t = _as_scalar(p, F)
    floor = _as_scalar(Emin_frac, F)
    if bool(torch.any(p_t <= 0.0).item()):
        raise ValueError("SIMP exponent p must be positive")
    if bool(torch.any((floor <= 0.0) | (floor > 1.0)).item()):
        raise ValueError("Emin_frac must lie in (0, 1]")
    gamma, gamma_prime = wang2014_gamma(
        rho,
        mode=gamma_mode,
        q=p_t if gamma_mode == "simp_heaviside" else q,
        beta0=beta0,
        eta0=eta0,
        return_derivative=True,
    )
    scale = floor + torch.pow(rho, p_t) * (1.0 - floor)
    # Endpoint convention (binding): rho == 1 must recover full-solid NH
    # values BIT FOR BIT — select, don't rely on floating-point algebra.
    scale = torch.where(rho == 1.0, torch.ones_like(scale), scale)
    scale_prime = p_t * torch.pow(rho, p_t - 1.0) * (1.0 - floor)
    identity = torch.eye(2, dtype=F.dtype, device=F.device)
    H = F - identity
    gamma_field = gamma.reshape(1, -1, 1, 1, 1)
    F_gamma = identity + gamma_field * H
    return H, F_gamma, gamma, gamma_prime, scale, scale_prime


def wang2014_energy_gp(
    F: torch.Tensor,
    rho: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_invalid: bool = False,
    return_bracket: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Interpolated energy ``s(rho) [Psi_NH(F_gamma) - Psi_L(gamma H) + Psi_L(H)]``.

    Gamma scales the DISPLACEMENT kinematics before the constitutive
    evaluation — not the stress afterwards. No autograd on the hot path.
    ``return_bracket`` also yields the unscaled bracket (the "unit" energy
    consumed by sensitivity code, D-ERR-2 naming lineage).
    """
    H, F_gamma, gamma, _, scale, _ = _wang2014_fields(
        F, rho, p, Emin_frac, gamma_mode, q, beta0, eta0
    )
    nonlinear, invalid = nh_energy_gp(F_gamma, mu, lam, return_invalid=True)
    gamma_field = gamma.reshape(1, -1, 1, 1, 1)
    linear_gamma = linear_energy_gp(gamma_field * H, mu, lam)
    linear_full = linear_energy_gp(H, mu, lam)
    bracket = nonlinear - linear_gamma + linear_full
    solid = gamma.reshape(1, -1, 1) == 1.0
    void = gamma.reshape(1, -1, 1) == 0.0
    bracket = torch.where(solid, nonlinear, bracket)
    bracket = torch.where(void, linear_full, bracket)
    energy = scale.reshape(1, -1, 1) * bracket
    if return_invalid and return_bracket:
        return energy, bracket, invalid
    if return_invalid:
        return energy, invalid
    if return_bracket:
        return energy, bracket
    return energy


def wang2014_first_piola(
    F: torch.Tensor,
    rho: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Exact analytic ``d(wang2014 energy)/dF``.

    Chain rule through ``F_gamma = I + gamma H`` gives the bracket
    ``gamma P_NH(F_gamma) - gamma P_L(gamma H) + P_L(H)``, scaled by s(rho).
    """
    H, F_gamma, gamma, _, scale, _ = _wang2014_fields(
        F, rho, p, Emin_frac, gamma_mode, q, beta0, eta0
    )
    nonlinear, invalid = nh_first_piola(F_gamma, mu, lam, return_invalid=True)
    gamma_field = gamma.reshape(1, -1, 1, 1, 1)
    linear_gamma = linear_first_piola(gamma_field * H, mu, lam)
    linear_full = linear_first_piola(H, mu, lam)
    bracket = gamma_field * nonlinear - gamma_field * linear_gamma + linear_full
    solid = gamma_field == 1.0
    void = gamma_field == 0.0
    bracket = torch.where(solid, nonlinear, bracket)
    bracket = torch.where(void, linear_full, bracket)
    P = scale.reshape(1, -1, 1, 1, 1) * bracket
    if return_invalid:
        return P, invalid
    return P


def wang2014_tangent(
    F: torch.Tensor,
    rho: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Exact packed tangent — both displacement-chain factors of gamma.

    ``d^2/dH^2`` brings ``gamma^2`` onto the NH and linear-gamma terms:
    ``A = s [ gamma^2 A_NH(F_gamma) - gamma^2 A_L + A_L ]`` (A_L constant).
    """
    _, F_gamma, gamma, _, scale, _ = _wang2014_fields(
        F, rho, p, Emin_frac, gamma_mode, q, beta0, eta0
    )
    nonlinear, invalid = nh_tangent(F_gamma, mu, lam, return_invalid=True)
    linear = linear_tangent(F, mu, lam)
    gamma_sq = gamma.reshape(1, -1, 1, 1, 1, 1, 1) ** 2
    bracket = gamma_sq * nonlinear - gamma_sq * linear + linear
    solid = gamma_sq == 1.0
    void = gamma_sq == 0.0
    bracket = torch.where(solid, nonlinear, bracket)
    bracket = torch.where(void, linear.expand_as(bracket), bracket)
    A = scale.reshape(1, -1, 1, 1, 1, 1, 1) * bracket
    if return_invalid:
        return A, invalid
    return A


def wang2014_energy_rho_derivative(
    F: torch.Tensor,
    rho: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Analytic ``d Psi_int / d rho`` — both density paths.

    ``d/drho = s'(rho) B + s(rho) gamma'(rho) H : (P_NH(F_gamma) - P_L(gamma H))``
    where ``B`` is the energy bracket. Kept explicit for the adjoint route —
    never autograd.
    """
    H, F_gamma, gamma, gamma_prime, scale, scale_prime = _wang2014_fields(
        F, rho, p, Emin_frac, gamma_mode, q, beta0, eta0
    )
    nonlinear_energy, invalid = nh_energy_gp(F_gamma, mu, lam, return_invalid=True)
    nonlinear_stress = nh_first_piola(F_gamma, mu, lam)
    gamma_field = gamma.reshape(1, -1, 1, 1, 1)
    linear_gamma_stress = linear_first_piola(gamma_field * H, mu, lam)
    linear_gamma_energy = linear_energy_gp(gamma_field * H, mu, lam)
    linear_full_energy = linear_energy_gp(H, mu, lam)
    bracket = nonlinear_energy - linear_gamma_energy + linear_full_energy
    solid = gamma.reshape(1, -1, 1) == 1.0
    void = gamma.reshape(1, -1, 1) == 0.0
    bracket = torch.where(solid, nonlinear_energy, bracket)
    bracket = torch.where(void, linear_full_energy, bracket)
    gamma_path = torch.sum(H * (nonlinear_stress - linear_gamma_stress), dim=(-2, -1))
    derivative = scale_prime.reshape(1, -1, 1) * bracket + (
        scale.reshape(1, -1, 1) * gamma_prime.reshape(1, -1, 1) * gamma_path
    )
    if return_invalid:
        return derivative, invalid
    return derivative


def wang2014_first_piola_rho_derivative(
    F: torch.Tensor,
    rho: torch.Tensor,
    mu: Scalar,
    lam: Scalar,
    p: Scalar,
    Emin_frac: Scalar,
    gamma_mode: str = "heaviside",
    q: float = 3.0,
    beta0: float = 500.0,
    eta0: float = 0.01,
    return_invalid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Analytic ``d P_int / d rho`` for adjoint residual-density products.

     Differentiating the stress bracket in rho:
     ``dP/drho = s' Q + s gamma' [P_NH + gamma A_NH:H - P_L(gamma H) - gamma A_L:H]``
     with ``Q`` the stress bracket. Consumed by the detached adjoint VJP
    .
    """
    H, F_gamma, gamma, gamma_prime, scale, scale_prime = _wang2014_fields(
        F, rho, p, Emin_frac, gamma_mode, q, beta0, eta0
    )
    nonlinear_stress, invalid = nh_first_piola(F_gamma, mu, lam, return_invalid=True)
    nonlinear_tangent = nh_tangent(F_gamma, mu, lam)
    gamma_field = gamma.reshape(1, -1, 1, 1, 1)
    linear_gamma_stress = linear_first_piola(gamma_field * H, mu, lam)
    linear_full_stress = linear_first_piola(H, mu, lam)
    linear_material = linear_tangent(F, mu, lam)
    nonlinear_increment = torch.einsum("...iJkL,...kL->...iJ", nonlinear_tangent, H)
    linear_increment = torch.einsum("iJkL,...kL->...iJ", linear_material, H)
    stress_bracket = (
        gamma_field * nonlinear_stress - gamma_field * linear_gamma_stress + linear_full_stress
    )
    solid = gamma_field == 1.0
    void = gamma_field == 0.0
    stress_bracket = torch.where(solid, nonlinear_stress, stress_bracket)
    stress_bracket = torch.where(void, linear_full_stress, stress_bracket)
    gamma_bracket = (
        nonlinear_stress
        + gamma_field * nonlinear_increment
        - linear_gamma_stress
        - gamma_field * linear_increment
    )
    derivative = scale_prime.reshape(1, -1, 1, 1, 1) * stress_bracket + (
        scale.reshape(1, -1, 1, 1, 1) * gamma_prime.reshape(1, -1, 1, 1, 1) * gamma_bracket
    )
    if return_invalid:
        return derivative, invalid
    return derivative
