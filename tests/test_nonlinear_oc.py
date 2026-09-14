"""Physical and sensitivity checks for the nonlinear density comparator."""

import numpy as np
import pytest
import torch

from hgto.fem.mesh.q4 import structured_q4
from hgto.linear2d.design import volume_density
from hgto.nonlinear.oc import NonlinearDensityMap, oc_update
from hgto.nonlinear.optimization import CaseSetup, operator, solve


def small_case():
    mesh = structured_q4(8, 4)
    mesh.coords[:, 0] *= 0.5
    mesh.coords[:, 1] *= 0.5
    # A nonuniform mesh makes the area-weighted filter check meaningful.
    mesh.coords[:, 0] = 4 * (mesh.coords[:, 0] / 4) ** 1.12
    left = np.flatnonzero(np.isclose(mesh.coords[:, 0], 0))
    right = np.flatnonzero(np.isclose(mesh.coords[:, 0], 4))
    fixed = np.sort(np.r_[2 * left, 2 * left + 1])
    force = np.zeros((mesh.n_nodes, 2))
    force[right, 1] = -1 / len(right)
    return CaseSetup(
        mesh,
        fixed,
        force,
        np.zeros(mesh.n_elements, bool),
        np.zeros(mesh.n_elements, bool),
        0.45,
        "small_nonlinear",
        {},
    )


def test_physical_density_map_matches_graph_map_with_cell_measures():
    torch.set_num_threads(1)
    setup = small_case()
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    rng = np.random.default_rng(17)
    x = rng.uniform(0.15, 0.8, setup.mesh.n_elements)
    rho, _ = mapping.physical(x, 3.0)
    z = torch.as_tensor(np.log(x / (1 - x)), dtype=torch.float64)
    graph_rho = volume_density(
        z,
        mapping.weights @ rho,
        mapping.filter.F,
        3.0,
        0.001,
        torch.as_tensor(setup.mesh.element_volumes(), dtype=torch.float64),
    )
    np.testing.assert_allclose(graph_rho.numpy(), rho, atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["terminal_work", "complementary_work"])
@pytest.mark.parametrize("gamma_mode", ["heaviside", "simp_heaviside"])
def test_nonlinear_filtered_adjoint_matches_directional_finite_difference(objective, gamma_mode):
    torch.set_num_threads(1)
    setup = small_case()
    setup.raw["nh_interpolation"] = dict(gamma_mode=gamma_mode, beta0=500.0, eta0=0.01)
    op = operator(setup, p=2.0)
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    rng = np.random.default_rng(29)
    # The revised case exercises the actual gamma transition and its rho**p
    # chain rule, rather than testing an already saturated solid region.
    x = (
        rng.uniform(0.13, 0.17, setup.mesh.n_elements)
        if gamma_mode == "simp_heaviside"
        else rng.uniform(0.38, 0.61, setup.mesh.n_elements)
    )
    direction = rng.normal(size=len(x))
    direction /= np.linalg.norm(direction)
    rho, dh = mapping.physical(x, 2.0)
    force = torch.as_tensor(setup.f, dtype=torch.float64) * (
        1e-5 if gamma_mode == "simp_heaviside" else 0.005
    )
    _, grad, _ = solve(op, torch.as_tensor(rho), force, "nh", True, 4, objective=objective)
    analytic = mapping.pullback(grad.numpy(), dh) @ direction
    eps = 1e-5
    values = []
    for sign in (-1, 1):
        trial_rho, _ = mapping.physical(x + sign * eps * direction, 2.0)
        value, _, _ = solve(
            op, torch.as_tensor(trial_rho), force, "nh", False, 4, objective=objective
        )
        values.append(value)
    finite = (values[1] - values[0]) / (2 * eps)
    assert analytic == pytest.approx(finite, rel=3e-5, abs=1e-9)


def test_nonlinear_oc_step_satisfies_volume_and_lowers_common_objective():
    torch.set_num_threads(1)
    setup = small_case()
    op = operator(setup, p=3.0)
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    x, rho = mapping.enforce_volume(np.full(setup.mesh.n_elements, 0.45), 2.0, 0.45)
    _, dh = mapping.physical(x, 2.0)
    force = torch.as_tensor(setup.f, dtype=torch.float64) * 0.005
    before, grad, _ = solve(
        op, torch.as_tensor(rho), force, "nh", True, 4, objective="complementary_work"
    )
    dc = mapping.pullback(grad.numpy(), dh)
    dv = mapping.pullback(mapping.weights, dh)
    updated, next_rho, info = oc_update(x, dc, dv, mapping, 0.45, 2.0, 0.05)
    after, _, details = solve(
        op, torch.as_tensor(next_rho), force, "nh", False, 4, objective="complementary_work"
    )
    assert mapping.weights @ next_rho == pytest.approx(0.45, abs=1e-10)
    assert after < before
    assert details["residual"] < 1e-8
    assert info["positive_gradient_count"] == 0


def test_material_positive_sensitivities_are_rejected_explicitly():
    setup = small_case()
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    x = np.full(setup.mesh.n_elements, 0.45)
    dc = -np.ones_like(x)
    dc[0] = 0.1
    with pytest.raises(ValueError, match="materially positive"):
        oc_update(x, dc, np.ones_like(x), mapping, 0.45, 1.0, 0.2)


def test_reciprocal_linear_branch_satisfies_mass_and_preserves_classic_path():
    setup = small_case()
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    x, _ = mapping.enforce_volume(np.full(setup.mesh.n_elements, 0.45), 2.0, 0.45)
    _, dh = mapping.physical(x, 2.0)
    dv = mapping.pullback(mapping.weights, dh)
    dc = -np.linspace(0.5, 2.0, len(x))
    strict = oc_update(x, dc, dv, mapping, 0.45, 2.0, 0.05)
    signed = oc_update(x, dc, dv, mapping, 0.45, 2.0, 0.05, mixed_sign_update="reciprocal_linear")
    np.testing.assert_array_equal(strict[0], signed[0])
    np.testing.assert_array_equal(strict[1], signed[1])
    dc[0] = 0.1
    candidate, rho, info = oc_update(
        x, dc, dv, mapping, 0.45, 2.0, 0.05, mixed_sign_update="reciprocal_linear"
    )
    assert candidate[0] == pytest.approx(x[0] - 0.05)
    assert mapping.weights @ rho == pytest.approx(0.45, abs=1e-10)
    assert info["positive_gradient_count"] == 1


def test_reciprocal_linear_branch_allows_signed_equality_multiplier():
    setup = small_case()
    mapping = NonlinearDensityMap(setup.mesh, 0.85)
    x, _ = mapping.enforce_volume(np.full(setup.mesh.n_elements, 0.45), 2.0, 0.45)
    dc = np.ones_like(x)
    dc[0] = -1
    candidate, rho, info = oc_update(
        x, dc, np.ones_like(x), mapping, 0.45, 2.0, 0.05, mixed_sign_update="reciprocal_linear"
    )
    assert mapping.weights @ rho == pytest.approx(0.45, abs=3e-12)
    assert np.max(np.abs(candidate - x)) <= 0.05 + 1e-12
    assert dc @ (candidate - x) < 0
    assert info["signed_multiplier_used"] and info["volume_multiplier"] < 0


def _controlled_acceptance_problem(monkeypatch, mode="overshoot"):
    """Isolate candidate acceptance with exact mass and a known overshoot.

    The physical solver and OC sensitivities are independently checked above.
    This controlled objective makes rejection and exhaustion deterministic.
    """
    import hgto.nonlinear.oc as module

    setup = small_case()
    mapping = NonlinearDensityMap(setup.mesh, 1.0)
    x0, rho0 = mapping.enforce_volume(np.full(setup.mesh.n_elements, 0.45), 1.0, 0.45)

    def proposal(x, dc, dv, mapping, target, beta, move, *unused):
        direction = np.zeros_like(x)
        direction[0], direction[-1] = 0.2, -0.2
        candidate, rho = mapping.enforce_volume(x + direction, beta, target)
        return candidate, rho, {}

    _, full_rho, _ = proposal(x0, None, None, mapping, 0.45, 1.0, 0.2)
    scale = full_rho[0] - rho0[0]
    assert scale > 0
    seen_volumes = []

    def controlled_state(op, rho, force, physics, gradient, *args, **kwargs):
        volume = float(mapping.weights @ rho.numpy())
        seen_volumes.append(volume)
        a = float((rho[0] - rho0[0]) / scale)
        if mode == "state_failure" and abs(a) > 1e-7:
            raise RuntimeError("Controlled inadmissible candidate")
        value = 1 + 4 * a * a - 3 * a if mode == "overshoot" else 1 + a * a
        return value, torch.full_like(rho, -1.0), {"residual": 0.0, "newton": 1}

    monkeypatch.setattr(module, "oc_update", proposal)
    monkeypatch.setattr(module, "solve", controlled_state)
    monkeypatch.setattr(module, "diagnostics", lambda details, spec: dict(details))
    return module, setup, rho0, seen_volumes


@pytest.mark.parametrize("acceptance", ["state_solvable", "fixed_parameter_descent"])
def test_fixed_parameter_descent_rejects_a_solved_overshoot(monkeypatch, tmp_path, acceptance):
    module, setup, _, volumes = _controlled_acceptance_problem(monkeypatch)
    config = module.NonlinearOCConfig(
        continuation_updates=0, max_updates=1, beta_final=1.0, acceptance=acceptance
    )
    result = module.run(setup.name, "nh", 0.005, tmp_path / "run", config, setup=setup)
    assert result["termination"] == "budget"
    assert not result["converged"]
    if acceptance == "fixed_parameter_descent":
        assert result["final"]["C"] < 1.0
        assert result["final"]["candidate_step_reductions"] >= 1
        assert result["objective_rejected_candidates"] >= 1
        assert result["final"]["descent_checked"]
    else:
        assert result["final"]["C"] > 1.0
        assert result["objective_rejected_candidates"] == 0
        assert not result["final"]["descent_checked"]
    assert result["failed_state_attempts"] == 0
    assert result["total_state_calls"] == 3 + result["objective_rejected_candidates"]
    assert result["successful_states"] == result["total_state_calls"]
    np.testing.assert_allclose(volumes, 0.45, atol=3e-12, rtol=0)


def test_descent_does_not_compare_different_continuation_parameters(monkeypatch, tmp_path):
    module, setup, _, volumes = _controlled_acceptance_problem(monkeypatch, "uphill")
    config = module.NonlinearOCConfig(
        continuation_updates=1, max_updates=1, beta_final=2.0, acceptance="fixed_parameter_descent"
    )
    result = module.run(setup.name, "nh", 0.005, tmp_path / "run", config, setup=setup)
    assert result["final"]["C"] > 1.0
    assert result["design_updates"] == 1
    assert not result["final"]["descent_checked"]
    assert result["rejected_candidates"] == 0
    assert result["total_state_calls"] == 3
    np.testing.assert_allclose(volumes, 0.45, atol=3e-12, rtol=0)


@pytest.mark.parametrize("mode", ["uphill", "state_failure"])
def test_exhausted_descent_preserves_last_accepted_state_without_false_convergence(
    monkeypatch, tmp_path, mode
):
    module, setup, rho0, volumes = _controlled_acceptance_problem(monkeypatch, mode)
    config = module.NonlinearOCConfig(
        continuation_updates=0,
        max_updates=1,
        beta_final=1.0,
        acceptance="fixed_parameter_descent",
        max_candidate_backtracks=2,
    )
    result = module.run(setup.name, "nh", 0.005, tmp_path / "run", config, setup=setup)
    assert result["termination"] == "stalled"
    assert not result["converged"]
    assert result["design_updates"] == 0
    assert result["optimization_state_evaluations"] == 1
    assert result["line_search_stop"]["attempted_iteration"] == 1
    assert result["final"]["iteration"] == 0
    assert result["final"]["C"] == pytest.approx(1.0)
    assert result["total_state_calls"] == 5
    assert result["rejected_candidates"] == 3
    assert result["failed_state_attempts"] == (3 if mode == "state_failure" else 0)
    assert result["objective_rejected_candidates"] == (3 if mode == "uphill" else 0)
    np.testing.assert_array_equal(np.load(tmp_path / "run/rho.npy"), rho0)
    np.testing.assert_allclose(volumes, 0.45, atol=3e-12, rtol=0)
