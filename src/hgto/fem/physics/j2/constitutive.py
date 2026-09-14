"""Batched small-strain associative J2 plasticity with isotropic hardening.

A closed-form radial return updates the plastic strain tensor and
accumulated equivalent plastic strain. The consistent tangent supports
incremental equilibrium and the transient adjoint. SIMP scales Young's
modulus, initial yield stress and hardening modulus together. Engineering
plane-strain input is embedded in the three-dimensional strain tensor."""

from __future__ import annotations

import torch

SQRT_TWO_THIRDS = (2.0 / 3.0) ** 0.5


def _field(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _identity_tensors(like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(eye, I_sym, I_vol) on the reference dtype/device."""
    eye = torch.eye(3, dtype=like.dtype, device=like.device)
    sym = 0.5 * (torch.einsum("ik,jl->ijkl", eye, eye) + torch.einsum("il,jk->ijkl", eye, eye))
    volumetric = torch.einsum("ij,kl->ijkl", eye, eye)
    return eye, sym, volumetric


def simp_scaled_j2_material(
    rho: torch.Tensor,
    E0: float,
    Emin: float,
    p: float,
    sigma_y0: float,
    H: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consistent SIMP weighting of the J2 material triple.

    One scale ``s(rho) = (Emin + rho^p (E0 - Emin)) / E0`` multiplies E,
    sigma_y0 and H together, so the return-mapped stress is homogeneous
    degree-1 in ``s`` and the yield strain is density-independent — the
    property both the SIMP sensitivity closed form and the void response
    depend on. Returns ``(E_e, sigma_y0_e, H_e)`` element fields.
    """
    scale = (Emin + torch.pow(rho, p) * (E0 - Emin)) / E0
    return E0 * scale, sigma_y0 * scale, H * scale


def j2_return_mapping(
    strain: torch.Tensor,
    plastic_strain_prev: torch.Tensor,
    alpha_prev: torch.Tensor,
    E,
    nu,
    sigma_y0,
    H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Radial return on ``(..., 3, 3)`` tensor strains.

    Returns ``(stress, plastic_strain, alpha, tangent, plastic_mask)``;
    ``tangent`` is the algorithmic tangent with trailing ``(3, 3, 3, 3)``.
    Material arguments broadcast over the batch (scalars or per-point
    fields).
    """
    if strain.shape[-2:] != (3, 3):
        raise ValueError("strain must have trailing shape (3, 3)")
    if plastic_strain_prev.shape[-2:] != (3, 3):
        raise ValueError("plastic_strain_prev must have trailing shape (3, 3)")
    eye, sym, volumetric = _identity_tensors(strain)
    deviatoric_projector = sym - volumetric / 3.0

    E_f = _field(E, strain)
    nu_f = _field(nu, strain)
    yield0 = _field(sigma_y0, strain)
    hardening = _field(H, strain)
    alpha_start = _field(alpha_prev, strain)

    mu = E_f / (2.0 * (1.0 + nu_f))
    kappa = E_f / (3.0 * (1.0 - 2.0 * nu_f))
    lam = kappa - 2.0 * mu / 3.0

    # elastic trial state
    elastic_strain = strain - plastic_strain_prev
    trace_e = elastic_strain.diagonal(dim1=-2, dim2=-1).sum(-1)
    trial = (
        lam[..., None, None] * trace_e[..., None, None] * eye
        + 2.0 * mu[..., None, None] * elastic_strain
    )
    trial_dev = trial - (trial.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0)[..., None, None] * eye
    trial_norm = torch.sqrt(torch.clamp((trial_dev * trial_dev).sum(dim=(-2, -1)), min=0.0))

    # yield check and closed-form plastic corrector (linear hardening)
    flow_stress = yield0 + hardening * alpha_start
    overshoot = trial_norm - SQRT_TWO_THIRDS * flow_stress
    plastic_mask = overshoot > 0.0
    norm_guard = torch.where(trial_norm > 0.0, trial_norm, torch.ones_like(trial_norm))
    flow_normal = trial_dev / norm_guard[..., None, None]
    dgamma = torch.clamp(overshoot, min=0.0) / (2.0 * mu + (2.0 / 3.0) * hardening)

    stress_dev = trial_dev - (2.0 * mu * dgamma)[..., None, None] * flow_normal
    stress = stress_dev + (kappa * trace_e)[..., None, None] * eye
    plastic_strain = plastic_strain_prev + dgamma[..., None, None] * flow_normal
    alpha = alpha_start + SQRT_TWO_THIRDS * dgamma

    # algorithmic tangent (S&H Box 3.2): elastic where the trial stays
    # inside the surface, radial-return-consistent otherwise
    theta = 1.0 - 2.0 * mu * dgamma / norm_guard
    theta_bar = 1.0 / (1.0 + hardening / (3.0 * mu)) - (1.0 - theta)
    nn = torch.einsum("...ij,...kl->...ijkl", flow_normal, flow_normal)
    expand = (...,) + (None,) * 4
    tangent_plastic = (
        kappa[expand] * volumetric
        + (2.0 * mu * theta)[expand] * deviatoric_projector
        - (2.0 * mu * theta_bar)[expand] * nn
    )
    tangent_elastic = kappa[expand] * volumetric + (2.0 * mu)[expand] * deviatoric_projector
    tangent = torch.where(plastic_mask[expand], tangent_plastic, tangent_elastic)
    return stress, plastic_strain, alpha, tangent, plastic_mask


def plane_strain_to_tensor(voigt: torch.Tensor) -> torch.Tensor:
    """Engineering plane-strain Voigt ``[exx, eyy, gxy]`` -> ``(..., 3, 3)``.

    ``eps_zz = 0`` (plane strain); engineering shear maps to the tensor
    component ``exy = gxy / 2``.
    """
    if voigt.shape[-1] != 3:
        raise ValueError("voigt strain must have trailing shape (3,)")
    exx, eyy = voigt[..., 0], voigt[..., 1]
    exy = 0.5 * voigt[..., 2]
    zero = torch.zeros_like(exx)
    return torch.stack(
        [
            torch.stack([exx, exy, zero], dim=-1),
            torch.stack([exy, eyy, zero], dim=-1),
            torch.stack([zero, zero, zero], dim=-1),
        ],
        dim=-2,
    )


def tensor_stress_to_plane_voigt(stress: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` stress -> in-plane Voigt ``[sxx, syy, sxy]``.

    ``sigma_zz`` is a plane-strain reaction and does no work against
    plane-strain kinematics; the in-plane components carry the residual.
    """
    if stress.shape[-2:] != (3, 3):
        raise ValueError("stress must have trailing shape (3, 3)")
    return torch.stack([stress[..., 0, 0], stress[..., 1, 1], stress[..., 0, 1]], dim=-1)
