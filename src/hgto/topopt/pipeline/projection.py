"""Tanh-Heaviside and exact-volume logistic density projections."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


# S has units of volume/logit.  Below this relative scale the implicit
# derivative is numerically undefined because all free sigmoids are saturated.
SATURATION_REL_THRESHOLD = 1.0e-14


def _heaviside_terms(
    density_bar: torch.Tensor,
    beta: float,
    eta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(density_bar):
        raise TypeError("density_bar must be a torch tensor")
    if density_bar.dtype != torch.float64:
        raise TypeError("Heaviside projection requires float64")
    if density_bar.numel() == 0 or not bool(torch.all(torch.isfinite(density_bar)).item()):
        raise ValueError("density_bar must be non-empty and finite")
    if not float(beta) > 0.0:
        raise ValueError("beta must be positive")
    if not 0.0 < float(eta) < 1.0:
        raise ValueError("eta must lie in (0, 1)")
    beta_t = density_bar.new_tensor(float(beta))
    eta_t = density_bar.new_tensor(float(eta))
    lower = torch.tanh(beta_t * eta_t)
    shifted = torch.tanh(beta_t * (density_bar - eta_t))
    denominator = lower + torch.tanh(beta_t * (1.0 - eta_t))
    return beta_t, shifted, denominator


def heaviside_projection(
    density_bar: torch.Tensor,
    beta: float,
    eta: float = 0.5,
) -> torch.Tensor:
    """Apply the normalized tanh-Heaviside density projection.

    The endpoints map exactly as ``H(0)=0`` and ``H(1)=1``.  This is the
    physical-field projection used by the min-volume three-field pipeline;
    it is distinct from the exact-volume logistic projection below.
    """
    _, shifted, denominator = _heaviside_terms(density_bar, beta, eta)
    lower = torch.tanh(density_bar.new_tensor(float(beta) * float(eta)))
    return (lower + shifted) / denominator


def heaviside_projection_derivative(
    density_bar: torch.Tensor,
    beta: float,
    eta: float = 0.5,
) -> torch.Tensor:
    """Return the pointwise derivative ``d H_beta / d density_bar``."""
    beta_t, shifted, denominator = _heaviside_terms(density_bar, beta, eta)
    return beta_t * (1.0 - shifted * shifted) / denominator


def _mask_like(mask: Optional[Any], reference: torch.Tensor, name: str) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    result = torch.as_tensor(mask, dtype=torch.bool, device=reference.device).reshape(-1)
    if result.shape != reference.shape:
        raise ValueError("{} must have shape (Ne,)".format(name))
    return result


class _VolumeProject(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        logits_bar: torch.Tensor,
        element_volume: torch.Tensor,
        target_volume: torch.Tensor,
        beta: float,
        rho_min: float,
        passive_solid: torch.Tensor,
        passive_void: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        free = ~(passive_solid | passive_void)
        free_volume = element_volume[free]
        solid_volume = torch.sum(element_volume[passive_solid])
        target = float(target_volume.item())
        minimum = float((solid_volume + float(rho_min) * torch.sum(free_volume)).item())
        maximum = float((solid_volume + torch.sum(free_volume)).item())
        scale = max(float(torch.sum(element_volume).item()), 1.0)
        feasibility_tol = 16.0 * torch.finfo(logits_bar.dtype).eps * scale
        if target < minimum - feasibility_tol or target > maximum + feasibility_tol:
            raise ValueError(
                "target volume is infeasible with passive masks: attainable [{:.17g}, {:.17g}]".format(
                    minimum, maximum
                )
            )
        if not bool(torch.any(free).item()):
            raise ValueError("volume projection has no free elements (S is saturated)")

        free_logits = logits_bar[free]

        def evaluated(lam_value: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
            sigmoid = torch.sigmoid(float(beta) * (free_logits - lam_value))
            free_rho = float(rho_min) + (1.0 - float(rho_min)) * sigmoid
            volume = solid_volume + torch.sum(free_volume * free_rho)
            return sigmoid, free_rho, float(volume.item())

        spread = max(1.0, 1.0 / float(beta))
        lower = float(torch.min(free_logits).item()) - spread
        upper = float(torch.max(free_logits).item()) + spread
        _, _, volume_lower = evaluated(lower)
        _, _, volume_upper = evaluated(upper)
        for _ in range(256):
            if volume_lower >= target:
                break
            spread *= 2.0
            lower -= spread
            _, _, volume_lower = evaluated(lower)
        for _ in range(256):
            if volume_upper <= target:
                break
            spread *= 2.0
            upper += spread
            _, _, volume_upper = evaluated(upper)
        if volume_lower < target or volume_upper > target:
            raise ValueError("target requires a saturated volume-projection endpoint")

        volume_tolerance = 1.0e-14 * max(abs(target), scale, 1.0)
        lam = 0.5 * (lower + upper)
        sigmoid, free_rho, volume = evaluated(lam)
        for _ in range(256):
            lam = 0.5 * (lower + upper)
            sigmoid, free_rho, volume = evaluated(lam)
            if abs(volume - target) <= volume_tolerance:
                break
            old_lower = lower
            old_upper = upper
            if volume > target:
                lower = lam
            else:
                upper = lam
            if lam == old_lower or lam == old_upper:
                break

        rho = torch.empty_like(logits_bar)
        rho[free] = free_rho
        rho[passive_solid] = 1.0  # fixed inside the projection.
        rho[passive_void] = 0.0
        q = torch.zeros_like(logits_bar)
        q[free] = (1.0 - float(rho_min)) * float(beta) * sigmoid * (1.0 - sigmoid)
        S = torch.sum(element_volume * q)
        saturation_threshold = SATURATION_REL_THRESHOLD * torch.sum(free_volume).clamp_min(1.0)
        if (not bool(torch.isfinite(S).item())) or float(S.item()) <= float(
            saturation_threshold.item()
        ):
            raise ValueError(
                "volume projection is hard-saturated: S <= {:.1e} * free volume".format(
                    SATURATION_REL_THRESHOLD
                )
            )
        if abs(volume - target) > 1.0e-12 * max(abs(target), 1.0):
            raise RuntimeError("volume projection bisection did not reach fp64 tolerance")

        ctx.save_for_backward(q, element_volume, free, S)
        return rho, logits_bar.new_tensor(lam)

    @staticmethod
    def backward(ctx: Any, grad_rho: torch.Tensor, grad_lam: Any) -> Tuple[Any, ...]:
        q, element_volume, free, S = ctx.saved_tensors
        # the lambda output is diagnostic and intentionally detached
        # from the design VJP.  Passive entries have q=0 and therefore no grad.
        weighted_cotangent = torch.sum(grad_rho * q)
        grad_logits = q * (grad_rho - element_volume * weighted_cotangent / S)
        grad_logits = torch.where(free, grad_logits, torch.zeros_like(grad_logits))
        return grad_logits, None, None, None, None, None, None


def volume_project(
    logits_bar: torch.Tensor,
    element_volume: Any,
    target_volume: Any,
    beta: float,
    rho_min: float,
    passive_solid: Optional[Any] = None,
    passive_void: Optional[Any] = None,
) -> Tuple[torch.Tensor, float]:
    """Project logits to the exact target material volume.

    ``target_volume`` is an absolute volume, not a fraction.  Saturation is
    rejected when ``S <= 1e-14 * max(free_volume, 1)`` because the IFT
    denominator is then numerically unusable.
    """
    if not torch.is_tensor(logits_bar):
        raise TypeError("logits_bar must be a torch tensor")
    if logits_bar.ndim != 1 or logits_bar.numel() == 0:
        raise ValueError("logits_bar must have shape (Ne,) and be non-empty")
    if logits_bar.dtype != torch.float64:
        raise TypeError("volume_project requires float64 logits")
    if not bool(torch.all(torch.isfinite(logits_bar)).item()):
        raise ValueError("logits_bar must be finite")
    volumes = torch.as_tensor(
        element_volume, dtype=logits_bar.dtype, device=logits_bar.device
    ).reshape(-1)
    if volumes.shape != logits_bar.shape or bool(torch.any(volumes <= 0.0).item()):
        raise ValueError("element_volume must be positive with shape (Ne,)")
    target = torch.as_tensor(
        target_volume, dtype=logits_bar.dtype, device=logits_bar.device
    ).reshape(-1)
    if target.numel() != 1 or not bool(torch.isfinite(target).item()):
        raise ValueError("target_volume must be a finite scalar")
    if float(beta) <= 0.0:
        raise ValueError("beta must be positive")
    if not 0.0 <= float(rho_min) < 1.0:
        raise ValueError("rho_min must lie in [0, 1)")
    solid = _mask_like(passive_solid, logits_bar, "passive_solid")
    void = _mask_like(passive_void, logits_bar, "passive_void")
    if bool(torch.any(solid & void).item()):
        raise ValueError("an element cannot be both passive solid and passive void")

    rho, lam_tensor = _VolumeProject.apply(
        logits_bar,
        volumes,
        target.reshape(()),
        float(beta),
        float(rho_min),
        solid,
        void,
    )
    return rho, float(lam_tensor.detach().item())
