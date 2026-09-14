import numpy as np
import pytest
from hgto.stopping import PhysicalStopping, StopConfig


def feed(s, objective=1.0, rho=0.4, **kwargs):
    args = dict(
        parameters=(3.0, 8.0),
        continuation_complete=True,
        volume_error=0.0,
        residual=1e-10,
        accepted=True,
    )
    args.update(kwargs)
    return s.observe(objective, np.array([rho]), **args)


def test_needs_final_parameters_objective_and_density_and_patience():
    s = PhysicalStopping(StopConfig(window=3, patience=2))
    for _ in range(15):
        assert not feed(s, continuation_complete=False)["converged"]
    assert not feed(s)["converged"]
    assert not feed(s)["converged"]
    assert not feed(s)["converged"]
    assert feed(s)["converged"]
    assert not feed(s, rho=0.42)["converged"]
    assert not feed(s, objective=1.01)["converged"]


@pytest.mark.parametrize(
    "change",
    [
        dict(parameters=(2.0, 4.0)),
        dict(accepted=False),
        dict(volume_error=1e-4),
        dict(residual=1e-5),
        dict(continuation_complete=False),
    ],
)
def test_parameter_change_rejection_or_invalid_state_clears_stability(change):
    s = PhysicalStopping(StopConfig(window=3, patience=2))
    for _ in range(5):
        last = feed(s)
    assert last["converged"]
    assert not feed(s, **change)["converged"]
    assert not feed(s)["converged"]


def test_cpu_checkpoint_continuation_matches_uninterrupted_adam(tmp_path):
    import torch
    from hgto.optimization import GraphOptimizerConfig, optimize_graph
    from hgto.linear2d.problems import make_problem
    from hgto.linear2d.physics import make_mesh, GraphElasticity

    p = make_problem("cantilever", nx=8, ny=4, radius=1.2)
    m = make_mesh(p)

    def run(limit, resume=None):
        c = GraphOptimizerConfig(
            n_freq=8,
            hidden_dim=16,
            device="cpu",
            stage_steps=(2, 2, 2, 2),
            max_updates=limit,
            early_stopping=False,
        )
        return optimize_graph(m, GraphElasticity(p, m, "cpu"), 0.5, 1.2, c, resume_from=resume)

    first = run(8)
    path = tmp_path / "resume.pt"
    torch.save(first["resume_checkpoint"], path)
    resumed = run(11, path)
    full = run(11)
    np.testing.assert_allclose(resumed["rho"], full["rho"], rtol=0, atol=1e-10)
    assert resumed["summary"]["design_updates"] == 11
