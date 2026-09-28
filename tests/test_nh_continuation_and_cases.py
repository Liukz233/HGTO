"""State continuation, Newton budgets and the finite-deformation L-bracket."""

import csv
import json

import numpy as np
import pytest
import torch

from hgto.nonlinear.oc import NonlinearOCConfig, run as run_oc
from hgto.nonlinear.optimization import case, operator, solve
from hgto.nonlinear.runner import validate_settings
from test_nonlinear_oc import small_case


def _nh(op, rho, force):
    return solve(op, rho, force, "nh", True, 6, objective="complementary_work")


def test_continued_state_reproduces_cold_reanalysis():
    torch.set_num_threads(1)
    setup = small_case()
    op = operator(setup)
    force = torch.tensor(setup.f) * 0.01
    rng = np.random.default_rng(3)
    first = torch.as_tensor(rng.uniform(0.3, 0.9, setup.mesh.n_elements))
    second = (first + 0.05 * torch.as_tensor(rng.uniform(-1, 1, len(first)))).clamp(0.2, 1)
    _, _, previous = _nh(op, first, force)
    cold = _nh(op, second, force)
    op.nh_state_start = previous["u"].clone()
    continued = _nh(op, second, force)
    assert cold[2]["state_start"] == "undeformed"
    assert continued[2]["state_start"] == "previous_state"
    assert continued[2]["newton"] < cold[2]["newton"]
    assert continued[0] == pytest.approx(cold[0], rel=1e-10)
    torch.testing.assert_close(continued[1], cold[1], rtol=1e-8, atol=1e-14)
    torch.testing.assert_close(continued[2]["u"], cold[2]["u"], rtol=1e-8, atol=1e-12)


def test_failed_continuation_falls_back_to_incremental_loading():
    torch.set_num_threads(1)
    setup = small_case()
    op = operator(setup)
    rho = torch.full((setup.mesh.n_elements,), 0.45)
    force = torch.tensor(setup.f) * 0.01
    cold = _nh(op, rho, force)
    op.nh_state_start = torch.full_like(cold[2]["u"], float("nan"))
    fallback = _nh(op, rho, force)
    assert fallback[2]["state_start"] == "undeformed"
    assert fallback[0] == pytest.approx(cold[0], rel=1e-12)


@pytest.mark.parametrize("state_continuation", [False, True])
def test_oc_state_continuation_keeps_final_cold_check(tmp_path, state_continuation):
    torch.set_num_threads(1)
    setup = small_case()
    setup.raw = {
        "L": 4.0,
        "volume_fraction": 0.45,
        "port_nodes": np.flatnonzero(setup.f[:, 1]).tolist(),
        "unit_direction": [0.0, -1.0],
    }
    config = NonlinearOCConfig(
        objective="complementary_work",
        continuation_updates=0,
        max_updates=3,
        load_steps=4,
        mixed_sign_update="reciprocal_linear",
        state_continuation=state_continuation,
    )
    record = run_oc("small", "nh", 0.01, tmp_path / "oc", config, setup=setup, spec=setup.raw)
    with (tmp_path / "oc" / "history.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    starts = [row["state_start"] for row in rows]
    assert starts[0] == "undeformed"
    if state_continuation:
        assert set(starts[1:]) == {"previous_state"}
    else:
        assert set(starts) == {"undeformed"}
    assert record["final_check_relative_error"] < 1e-9
    assert record["final_check"]["state_start"] == "undeformed"


def test_l_bracket_case_geometry_supports_and_load():
    setup, spec = case("lbracket_nh")
    mesh = setup.mesh
    assert (mesh.n_elements, mesh.n_nodes) == (3072, 3201)
    centroids = mesh.element_centroids()
    assert not np.any((centroids[:, 0] > 8) & (centroids[:, 1] > 8))
    assert mesh.element_volumes().sum() == pytest.approx(192.0, rel=1e-14)
    assert spec["mesh_area"] == pytest.approx(192.0, rel=1e-14)
    assert spec["removed_region"] == [8.0, 8.0, 16.0, 16.0]
    np.testing.assert_allclose(setup.f.sum(0), [0.0, -1.0], atol=1e-15)
    ports = mesh.coords[np.any(setup.f != 0, axis=1)]
    assert np.allclose(ports[:, 0], 16.0) and ports[:, 1].min() == pytest.approx(6.5)
    assert ports[:, 1].max() == pytest.approx(8.0)
    fixed_nodes = np.unique(setup.fixed_dofs // 2)
    assert np.allclose(mesh.coords[fixed_nodes, 1], 16.0)
    assert mesh.coords[fixed_nodes, 0].max() == pytest.approx(8.0)
    assert len(setup.fixed_dofs) == 2 * len(fixed_nodes)
    assert spec["nh_interpolation"]["gamma_mode"] == "simp_heaviside"


def test_newton_budgets_are_recorded_and_applied():
    budgets = {
        "max_newton": 40,
        "adaptive_budgets": {
            "max_subdivisions": 6,
            "max_failed_attempts": 4,
            "max_extra_increments": 32,
        },
    }
    setup, spec = case("cantilever_nh", nh_solver=budgets)
    assert spec["nh_solver"] == budgets
    op = operator(setup)
    assert op.nh_max_newton == 40
    assert op.nh_adaptive_budgets == budgets["adaptive_budgets"]
    for invalid in ({"max_newton": 0}, {"max_iterations": 40}, {"adaptive_budgets": {"x": 1}}):
        with pytest.raises(ValueError):
            case("cantilever_nh", nh_solver=invalid)
    with pytest.raises(ValueError):
        case("connection_j2", nh_solver=budgets)


def _settings(**case_options):
    return {
        "category": "nonlinear",
        "method": "hgto",
        "case": {"name": "lbracket_nh", "physics": "nh", "load": 0.04, **case_options},
        "hgto": {"device": "cpu", "state_continuation": True, "beta_ramp_base": 8},
    }


def test_runner_accepts_l_bracket_and_validates_case_options():
    validate_settings(_settings(nh_solver={"max_newton": 40}))
    for invalid in (
        _settings(hole_geometry="boundary_fitted"),
        _settings(nh_solver={"max_newton": -1}),
        _settings(refine=0),
    ):
        with pytest.raises(ValueError):
            validate_settings(invalid)
    wrong = _settings()
    wrong["case"]["physics"] = "j2"
    with pytest.raises(ValueError, match="do not match"):
        validate_settings(wrong)


def test_paper_nonlinear_configurations_share_problem_settings():
    from pathlib import Path

    from hgto.cli import load_settings

    root = Path(__file__).resolve().parents[1] / "configs" / "nonlinear"
    for path in sorted(root.glob("*.yaml")):
        hgto = load_settings(path)
        oc = load_settings(path, method="oc")
        assert hgto["case"] == oc["case"]
        assert oc["oc"]["device"] == "cpu" and oc["oc"]["tangent_backend"] == "pypardiso"
        assert hgto["hgto"]["state_continuation"] and oc["oc"]["state_continuation"]
        assert hgto["hgto"]["beta_final"] == oc["oc"]["beta_final"] == 32
        assert json.dumps(hgto["case"]["nh_solver"], sort_keys=True) == json.dumps(
            {
                "adaptive_budgets": {
                    "max_extra_increments": 32,
                    "max_failed_attempts": 4,
                    "max_subdivisions": 6,
                },
                "max_newton": 40,
            },
            sort_keys=True,
        )
