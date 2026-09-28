"""Stopping certificates must survive tiny optimizer steps and retries."""

import csv
import numpy as np
import pytest
import torch
from hgto.nonlinear.stationarity import projected_gradient


def test_projected_stationarity_is_zero_at_constrained_kkt_and_scale_invariant():
    x = np.array([0.3, 0.7])
    w = np.array([0.25, 0.75])
    dv = w.copy()
    d, info = projected_gradient(x, -2 * dv, dv, w, 3.0)
    assert info["projected_gradient_inf"] < 1e-11
    dc = np.array([-0.2, -1.8])
    d, info = projected_gradient(x, dc, dv, w, 3.0)
    assert info["projected_gradient_inf"] > 0.1
    assert dc @ d < 0 and abs(dv @ d) < 1e-12
    same, _ = projected_gradient(x, dc * 1e-9, dv, w, 3e-9)
    np.testing.assert_allclose(d, same, atol=1e-12)


def test_projected_stationarity_respects_active_bounds():
    x = np.array([1e-9, 1 - 1e-9])
    w = np.array([0.5, 0.5])
    _, info = projected_gradient(x, np.array([1.0, -1.0]), w, w, 1.0)
    assert info["projected_gradient_inf"] < 1e-12


def test_graph_backtracking_does_not_accumulate_old_gradients(monkeypatch, tmp_path):
    import hgto.nonlinear.optimization as mod
    from hgto.fem.mesh.q4 import structured_q4
    from hgto.topopt.pipeline.filter import DensityFilter

    torch.set_default_dtype(torch.float64)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z = torch.nn.Parameter(torch.tensor([0.2, -0.2]))

        def forward(self, *unused):
            return self.z

    net = Net()
    mesh = structured_q4(2, 1)
    f = np.zeros((mesh.n_nodes, 2))
    f[-1, 1] = -1
    setup = mod.CaseSetup(
        mesh,
        np.array([0, 1]),
        f,
        np.zeros(2, bool),
        np.zeros(2, bool),
        0.45,
        "controlled",
        {"L": 2.0, "volume_fraction": 0.45},
    )
    monkeypatch.setattr(
        mod, "graph_network", lambda *args, **kwargs: (net, {"coords": None, "edge_index": None})
    )
    monkeypatch.setattr(mod, "diagnostics", lambda *args: {"residual": 0.0})
    evaluations = []
    filt = DensityFilter(mesh, 1.0)
    vol = torch.tensor(mesh.element_volumes())

    def fake_solve(op, rho, *args, **kwargs):
        C = 1 + float((rho[0] - 0.25) ** 2)
        g = torch.tensor([2 * (rho[0] - 0.25), 0.0])
        zz = net.z.detach().clone().requires_grad_()
        rr = mod.density_field(zz, filt, vol, 0.45 * vol.sum(), 1.0, 0.001)
        grad = torch.autograd.grad(rr, zz, g / C)[0]
        evaluations.append((C, float(grad.norm())))
        return C, g, {"residual": 0.0}

    monkeypatch.setattr(mod, "solve", fake_solve)
    mod.run(
        "controlled",
        "linear",
        1.0,
        tmp_path,
        steps=1,
        beta_final=1.0,
        max_updates=12,
        early_stopping=False,
        learning_rate=8.0,
        final_learning_rate=8.0,
        minimum_learning_rate=8.0,
        device="cpu",
        setup=setup,
        spec=setup.raw,
        backtracking=True,
    )
    rows = list(csv.DictReader((tmp_path / "history.csv").open()))
    assert any(int(row["candidate_step_reductions"]) > 0 for row in rows)
    for row in rows:
        expected = next(norm for value, norm in evaluations if abs(value - float(row["C"])) < 1e-13)
        assert float(row["parameter_gradient_norm"]) == pytest.approx(expected, rel=1e-8, abs=1e-12)
    record = __import__("json").loads((tmp_path / "record.json").read_text())
    assert not record["converged"]


def test_adaptive_load_matches_fixed_ramp_when_no_cutback_is_needed():
    from hgto.nonlinear.optimization import operator, solve
    from test_nonlinear_oc import small_case

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    setup = small_case()
    op = operator(setup)
    rho = torch.full((setup.mesh.n_elements,), 0.45)
    force = torch.tensor(setup.f) * 0.005
    fixed = solve(op, rho, force, "nh", True, 4, objective="complementary_work")
    op.adaptive_nh_load = True
    adaptive = solve(op, rho, force, "nh", True, 4, objective="complementary_work")
    assert adaptive[0] == pytest.approx(fixed[0], rel=1e-12)
    torch.testing.assert_close(adaptive[1], fixed[1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        adaptive[2]["history_u"], fixed[2]["history_u"], rtol=1e-10, atol=1e-12
    )
    assert adaptive[2]["load_cutback_failures"] == 0
    assert adaptive[2]["load_increments"] == 4


def test_adaptive_load_cuts_back_from_last_success_and_records_stations(monkeypatch):
    import hgto.fem.physics.neohookean.adaptive as mod
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class State:
        u: object
        residual_rel: object
        newton_iterations: object
        pcg_iterations: object
        iterations: object
        backtracks: object
        indefinite_fallbacks: object
        inexact_directions: object
        load_factors: object = None
        ramp_newton_iterations: object = None
        ramp_residual_rel: object = None
        ramp_u: object = None
        t_state_total: float = 0.0

    attempts = []

    def fake(operator, rho, force, *, u0, **kwargs):
        left, right = float(u0.max()), float(force.max())
        attempts.append((left, right))
        if right - left > 0.3:
            raise RuntimeError("controlled oversized load step")
        one = torch.tensor([1])
        zero = torch.tensor([0])
        return State(force.clone(), torch.tensor([1e-12]), one, one, one, zero, zero, zero)

    monkeypatch.setattr(mod, "solve_nh_state", fake)
    op = SimpleNamespace(dtype=torch.float64, device=torch.device("cpu"))
    result = mod.solve_nh_adaptive(op, torch.ones(1), torch.ones(1, 1, 2), n_ramp=2)
    assert attempts == [(0.0, 0.5), (0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (0.5, 0.75), (0.75, 1.0)]
    assert result.adaptive_failed_attempts == 2
    assert result.adaptive_successful_increments == 4
    torch.testing.assert_close(result.ramp_u[0, :, 0, 0], torch.tensor([0.5, 1.0]))
    assert result.ramp_newton_iterations.tolist() == [[2, 2]]
    with pytest.raises(RuntimeError, match="oversized"):
        mod.solve_nh_adaptive(op, torch.ones(1), torch.ones(1, 1, 2), n_ramp=2, max_subdivisions=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nonlinear_oc_cuda_state_matches_cpu_one_update(tmp_path):
    from hgto.nonlinear.oc import run, NonlinearOCConfig
    from test_nonlinear_oc import small_case

    setup = small_case()
    spec = {
        "L": 4.0,
        "volume_fraction": 0.45,
        "port_nodes": np.flatnonzero(setup.f[:, 1]).tolist(),
        "unit_direction": [0.0, -1.0],
    }
    setup.raw = spec
    base = dict(
        objective="complementary_work",
        continuation_updates=0,
        max_updates=1,
        load_steps=4,
        mixed_sign_update="reciprocal_linear",
    )
    cpu = run(
        "small", "nh", 0.005, tmp_path / "cpu", NonlinearOCConfig(**base), setup=setup, spec=spec
    )
    gpu = run(
        "small",
        "nh",
        0.005,
        tmp_path / "gpu",
        NonlinearOCConfig(**base, device="cuda:0", tangent_backend="cudss"),
        setup=setup,
        spec=spec,
    )
    assert gpu["final"]["C"] == pytest.approx(cpu["final"]["C"], rel=1e-10)
    np.testing.assert_allclose(
        np.load(tmp_path / "gpu/rho.npy"), np.load(tmp_path / "cpu/rho.npy"), rtol=1e-9, atol=1e-11
    )


def test_nh_evaluator_restores_saved_refinement_and_adaptive_policy(monkeypatch, tmp_path):
    import json
    import hgto.nonlinear.evaluation as mod
    from test_nonlinear_oc import small_case

    setup = small_case()
    ports = np.flatnonzero(setup.f[:, 1]).tolist()
    spec = {
        "case": "cantilever_nh",
        "L": 4.0,
        "volume_fraction": 0.45,
        "port_nodes": ports,
        "unit_direction": [0.0, -1.0],
        "fixed_dofs": setup.fixed_dofs.tolist(),
    }
    calls = []

    def fake_case(name, refine, nh_transition_beta):
        calls.append(refine)
        return setup, spec

    monkeypatch.setattr(mod, "case", fake_case)
    protocol = dict(spec, nx=192, force_resultant=0.00125, load_steps=3, adaptive_load=True)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for name, value in [
        ("coords", setup.mesh.coords),
        ("econn", setup.mesh.econn),
        ("unit_force", setup.f),
        ("rho", np.full(setup.mesh.n_elements, 0.45)),
    ]:
        np.save(tmp_path / (name + ".npy"), value)

    def fake_solve(op, rho, force, physics, gradient, steps, **kwargs):
        assert op.adaptive_nh_load and steps == 3
        return (
            1.0,
            None,
            {
                "u": torch.zeros(1, setup.mesh.n_nodes, 2),
                "history_u": torch.zeros(3, setup.mesh.n_nodes, 2),
                "residual": 0.0,
                "terminal_work": 0.0,
            },
        )

    monkeypatch.setattr(mod, "solve", fake_solve)
    result = mod.evaluate_nh(tmp_path)
    assert calls == [2] and result["adaptive_load"]


def test_oc_wall_budget_preserves_last_accepted_design(monkeypatch, tmp_path):
    from test_nonlinear_oc import _controlled_acceptance_problem

    module, setup, rho0, _ = _controlled_acceptance_problem(monkeypatch)
    elapsed = [0.0]
    original = module.solve

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        elapsed[0] += 1.0
        return result

    monkeypatch.setattr(module, "solve", delayed)
    monkeypatch.setattr(module.time, "perf_counter", lambda: elapsed[0])
    cfg = module.NonlinearOCConfig(
        continuation_updates=0,
        max_updates=10,
        beta_final=1.0,
        acceptance="fixed_parameter_descent",
        max_wall_seconds=0.5,
    )
    result = module.run(setup.name, "nh", 0.005, tmp_path / "run", cfg, setup=setup)
    assert result["termination"] == "wall_time_budget" and not result["converged"]
    assert result["design_updates"] == 0 and result["total_state_calls"] == 2
    np.testing.assert_array_equal(np.load(tmp_path / "run/rho.npy"), rho0)
    assert (tmp_path / "run/last_accepted_design.npy").exists()
