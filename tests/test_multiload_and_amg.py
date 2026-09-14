"""Independent physics and exact-volume checks for the extension."""

import numpy as np
import pytest
import torch
from hgto.linear2d.problems import make_problem
from hgto.linear2d.physics import make_mesh
from hgto.fem import MechanicsOperator
from hgto.reference import ScikitFEMElasticity
from hgto.linear2d.design import volume_density
from hgto.nonlinear.optimization import case

torch.set_num_threads(2)


def test_bridge_separate_rhs_and_gradient():
    problem = make_problem("bridge_multiload", nx=48, ny=16, radius=2.0, volume=0.4)
    f = problem.forces
    assert f.shape[1] == 3 and np.linalg.matrix_rank(f) == 3
    np.testing.assert_allclose(f[1::2].sum(0), -np.ones(3) / np.sqrt(3.0))
    np.testing.assert_allclose(f[0::2], 0.0)
    rho = np.linspace(0.25, 0.8, problem.n_elements)
    fem = ScikitFEMElasticity.from_problem(problem, solver="scipy")
    c, g, u = fem.evaluate(rho)
    # Reusing one matrix for three RHS must equal three independent energies,
    # and differ from the physically different simultaneous summed load.
    separate = sum(float(f[:, k] @ u[:, k]) for k in range(3))
    assert c == pytest.approx(separate, rel=1e-12)
    simultaneous = float(f.sum(1) @ u.sum(1))
    assert not np.isclose(c, simultaneous, rtol=0.01)
    direction = np.random.default_rng(1).normal(size=len(rho))
    direction /= np.linalg.norm(direction)
    h = 1e-5
    plus = fem.evaluate(rho + h * direction)[0]
    minus = fem.evaluate(rho - h * direction)[0]
    assert (plus - minus) / (2 * h) == pytest.approx(float(g @ direction), rel=2e-5)
    from hgto.linear2d.physics import GraphElasticity

    graph = GraphElasticity(problem, make_mesh(problem), device="cpu")
    gc, gg = graph.evaluate(torch.from_numpy(rho))
    assert gc == pytest.approx(c, rel=1e-7)
    np.testing.assert_allclose(np.asarray(gg), g, rtol=2e-5, atol=1e-6)
    fem.close()


def test_cached_transpose_preserves_weighted_volume_gradient():
    n = 9
    a = torch.eye(n, dtype=torch.float64) + 0.2 * torch.ones((n, n), dtype=torch.float64)
    a = a / a.sum(1)[:, None]
    a = a.to_sparse().coalesce()
    z = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, requires_grad=True)
    weights = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
    fn = lambda x: volume_density(x, 0.4, a, 8.0, 0.001, weights)
    assert torch.autograd.gradcheck(fn, (z,), eps=1e-5, atol=1e-5, rtol=1e-4)
    assert float((fn(z) * weights).sum() / weights.sum()) == pytest.approx(0.4, abs=1e-12)


def test_second_nh_geometry_and_common_material():
    bridge, spec = case("bridge_nh")
    cant, cantspec = case("cantilever_nh")
    assert spec["nh_interpolation"] == cantspec["nh_interpolation"]
    assert spec["mesh_elements"] == 96 * 32
    np.testing.assert_allclose(bridge.f.sum(0), [0.0, -1.0])
    fixed_nodes = np.unique(bridge.fixed_dofs // 2)
    assert set(np.unique(bridge.mesh.coords[fixed_nodes, 0])) == {0.0, 24.0}
    assert np.allclose(bridge.mesh.coords[spec["port_nodes"], 1], 8.0)


def test_unstructured_amg_matches_reference_and_reuse():
    from hgto.fem.mesh.q4 import distorted_q4

    mesh = distorted_q4(12, 8, amplitude=0.1, seed=2)
    nodes = np.flatnonzero(np.isclose(mesh.coords[:, 0], 0))
    fixed = (2 * nodes[:, None] + np.arange(2)).ravel()
    f = np.zeros((mesh.n_dof, 1))
    load = np.argmax(mesh.coords[:, 0] + 0.001 * mesh.coords[:, 1])
    f[2 * load + 1] = -1.0
    op = MechanicsOperator(
        mesh, fixed, dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0), device="cpu", preconditioner="amgcg"
    )
    op.preconditioner_reuse_steps = 10
    op.preconditioner_reuse_ratio = 2.0
    fem = ScikitFEMElasticity(mesh.coords, mesh.econn, fixed, f, solver="scipy")
    for delta in [0.0, 0.005]:
        rho = torch.linspace(0.3, 0.8, mesh.n_elements, dtype=torch.float64) + delta
        state = op.solve_state(
            rho,
            torch.from_numpy(f.T).reshape(1, -1, 2),
            rtol=1e-9,
            max_iter=2000,
            allow_direct_fallback=False,
        )
        c, g, u = fem.evaluate(rho.numpy())
        assert float(state.compliance) == pytest.approx(c, rel=1e-8)
        np.testing.assert_allclose(
            op.compliance_sensitivity(rho, state).numpy(), g, rtol=2e-6, atol=1e-5
        )
        assert float(state.residual_rel.max()) <= 1e-9
    assert op.mg_hierarchy_reuses == 1
    fem.close()
