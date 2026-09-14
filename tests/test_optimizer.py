"""Physical checks for fixed-budget Adam and common final mechanics."""

import numpy as np
import pytest
import torch
from hgto.optimization import GraphOptimizerConfig, optimize_graph
from hgto.linear2d.problems import make_problem
from hgto.linear2d.physics import make_mesh, GraphElasticity
from hgto.reference import ScikitFEMElasticity


def test_tensor_material_schedule_and_final_common_physics():
    torch.set_num_threads(1)
    problem = make_problem("cantilever", nx=8, ny=4, radius=1.2)
    mesh = make_mesh(problem)
    physics = GraphElasticity(problem, mesh, "cpu")
    cfg = GraphOptimizerConfig(
        n_freq=8,
        hidden_dim=16,
        device="cpu",
        stage_steps=(2, 2, 2, 2),
        penalties=(1.0, 2.0, 3.0, 3.0),
        max_updates=8,
    )
    result = optimize_graph(mesh, physics, 0.5, 1.2, cfg)
    assert float(physics.operator.p) == 3.0
    assert set(row["penalty"] for row in result["history"]) == {1.0, 2.0, 3.0}
    assert result["rho"].mean() == pytest.approx(0.5, abs=1e-12)
    library = ScikitFEMElasticity.from_problem(problem, solver="scipy")
    value, _, _ = library.evaluate(result["rho"])
    assert result["summary"]["C_raw"] == pytest.approx(value, rel=1e-7)
    assert [r["phase"] for r in result["history"]] == ["adam"] * 8 + ["final"]
    assert result["summary"]["design_updates"] == 8
    assert result["summary"]["optimizer"] == "Adam"
    assert np.all(np.isfinite(result["rho"]))
    library.close()


def test_refinement_cannot_be_enabled_in_current_protocol():
    with pytest.raises(TypeError, match="lbfgs_steps"):
        optimize_graph(None, None, 0.5, 1.0, GraphOptimizerConfig(lbfgs_steps=1))
