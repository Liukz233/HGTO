"""Independent derivative checks for the stiffness-based void interpolation."""

import numpy as np
import pytest
import torch
from hgto.fem.physics.neohookean.constitutive import (
    wang2014_gamma,
    wang2014_energy_gp,
    wang2014_first_piola,
    wang2014_energy_rho_derivative,
    wang2014_first_piola_rho_derivative,
)


def test_switch_tracks_current_material_penalty():
    for p in (1.0, 2.0, 3.0):
        rho = torch.tensor([0.01 ** (1 / p)], dtype=torch.float64)
        gamma, slope = wang2014_gamma(rho, mode="simp_heaviside", q=p, return_derivative=True)
        assert abs(float(gamma) - 0.5) < 3e-5
        eps = 1e-7
        finite = (
            wang2014_gamma(rho + eps, mode="simp_heaviside", q=p)
            - wang2014_gamma(rho - eps, mode="simp_heaviside", q=p)
        ) / (2 * eps)
        torch.testing.assert_close(slope, finite, rtol=2e-8, atol=1e-8)


@pytest.mark.parametrize("beta0", [20.0, 100.0, 500.0])
def test_full_energy_and_stress_density_chains(beta0):
    torch.manual_seed(781)
    for p in (1.7, 3.0):
        cutoff = 0.01 ** (1 / p)
        rho = torch.tensor(
            [0.001, 0.75 * cutoff, 0.95 * cutoff, cutoff, 1.05 * cutoff, 1.3 * cutoff, 0.6, 0.9999],
            dtype=torch.float64,
            requires_grad=True,
        )
        F = (
            torch.eye(2, dtype=torch.float64)
            + 0.15 * torch.randn(1, 8, 4, 2, 2, dtype=torch.float64)
        ).requires_grad_()
        args = dict(
            mu=1 / 2.6,
            lam=0.3 / 0.91,
            p=p,
            Emin_frac=1e-6,
            gamma_mode="simp_heaviside",
            beta0=beta0,
        )
        energy = wang2014_energy_gp(F, rho, **args)
        stress_ad, rho_ad = torch.autograd.grad(energy.sum(), (F, rho), create_graph=True)
        stress = wang2014_first_piola(F, rho, **args)
        gradient = wang2014_energy_rho_derivative(F, rho, **args).sum((0, 2))
        torch.testing.assert_close(stress, stress_ad, rtol=2e-10, atol=1e-12)
        torch.testing.assert_close(gradient, rho_ad, rtol=2e-10, atol=1e-12)
        direction = torch.randn_like(F)
        stress_vjp = torch.autograd.grad((stress_ad * direction).sum(), rho)[0]
        analytic = (wang2014_first_piola_rho_derivative(F, rho, **args) * direction).sum(
            (0, 2, 3, 4)
        )
        torch.testing.assert_close(analytic, stress_vjp, rtol=3e-9, atol=2e-11)


@pytest.mark.parametrize("beta0", [20.0, 100.0, 500.0])
def test_solid_endpoint_uses_one_sided_density_slope(beta0):
    # The full-solid energy takes an exact endpoint branch. Its physical
    # density derivative is the limit from below, not AD through that branch.
    rho = torch.ones(1, dtype=torch.float64)
    F = torch.tensor([[[[[1.1, 0.12], [0.04, 0.94]]]]], dtype=torch.float64)
    args = dict(
        mu=1 / 2.6, lam=0.3 / 0.91, p=3.0, Emin_frac=1e-6, gamma_mode="simp_heaviside", beta0=beta0
    )
    h = 1e-7
    difference = (wang2014_energy_gp(F, rho, **args) - wang2014_energy_gp(F, rho - h, **args)) / h
    analytic = wang2014_energy_rho_derivative(F, rho, **args)
    torch.testing.assert_close(analytic, difference, rtol=3e-7, atol=1e-10)


def test_new_case_declares_model_and_volume_initialization():
    from hgto.nonlinear.optimization import case, operator

    setup, spec = case("cantilever_nh")
    op = operator(setup)
    assert op.nh_interpolation == spec["nh_interpolation"]
    assert spec["nh_interpolation"]["gamma_mode"] == "simp_heaviside"
    np.testing.assert_allclose(setup.f.sum(0), [0.0, -1.0], atol=1e-15)
    assert setup.mesh.n_elements == 2304


def test_transition_override_preserves_geometry_load_and_solid_model():
    from hgto.nonlinear.optimization import case, operator

    original, _ = case("cantilever_nh")
    setup, spec = case("cantilever_nh", nh_transition_beta=100.0)
    np.testing.assert_array_equal(setup.mesh.coords, original.mesh.coords)
    np.testing.assert_array_equal(setup.f, original.f)
    assert operator(setup).nh_interpolation["beta0"] == 100.0
    assert spec["nh_interpolation"]["eta0"] == 0.01
    with pytest.raises(ValueError):
        case("cantilever_nh", nh_transition_beta=-1)
