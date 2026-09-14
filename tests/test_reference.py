"""Independent library FEM agreement, geometry and sensitivity regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

pytest.importorskip("skfem")

from hgto.fem import MechanicsOperator
from hgto.fem.mesh import structured_q4, structured_hex8
from hgto.fem.mesh.q4 import distorted_q4
from hgto.fem.mesh.hex8 import distorted_hex8
from hgto.reference import ScikitFEMElasticity
from hgto.reference.oc import OCConfig, optimize_oc
from hgto.linear2d.problems import make_problem
from hgto.linear2d.reference import Elasticity, q4_stiffness


@pytest.mark.parametrize("family", ["cantilever", "mbb", "multiload", "l_bracket"])
def test_q4_matches_existing_regular_mesh_oracle(family):
    problem = make_problem(family, 12, 6)
    rho = np.linspace(0.25, 0.85, problem.n_elements)
    library = ScikitFEMElasticity.from_problem(problem, solver="scipy")
    actual = library.evaluate(rho)
    expected = Elasticity(problem).evaluate(rho)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-8)
    assert library.last_residual < 1e-10
    assert library.metadata()["library"] == "scikit-fem"


def test_unequal_q4_uses_every_elements_own_geometry_and_thickness():
    mesh = distorted_q4(8, 4, amplitude=0.2, seed=5, thickness=0.7)
    mesh.coords[:, 0] *= 1.0 + 0.06 * mesh.coords[:, 0]
    force = np.zeros((mesh.n_dof, 2))
    force[2 * mesh.node_id(8, 2) + 1, 0] = -1.0
    force[2 * mesh.node_id(8, 4), 1] = 0.35
    fixed = (2 * mesh.left_edge_nodes()[:, None] + np.arange(2)).ravel()
    rho = np.linspace(0.3, 0.9, mesh.n_elements)
    library = ScikitFEMElasticity(
        mesh.coords, mesh.econn, fixed, force, thickness=0.7, solver="scipy"
    )
    c, g, u = library.evaluate(rho)
    # The former oracle caches only the first element's matrix and is invalid
    # on unequal cells; assemble each element separately for this regression.
    local = np.array([0.7 * q4_stiffness(mesh.coords[t]) for t in mesh.econn])
    dofs = (2 * mesh.econn[..., None] + np.arange(2)).reshape(-1, 8)
    modulus = 1e-6 + (1 - 1e-6) * rho**3
    matrix = sparse.coo_matrix(
        (
            (local * modulus[:, None, None]).ravel(),
            (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel()),
        ),
        shape=(mesh.n_dof, mesh.n_dof),
    ).tocsc()
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    expected = np.zeros_like(force)
    expected[free] = splu(matrix[free][:, free]).solve(force[free])
    np.testing.assert_allclose(u, expected, rtol=2e-9, atol=1e-8)
    assert c == pytest.approx(float((force * expected).sum()), rel=1e-10)
    np.testing.assert_allclose(
        library.element_volumes, mesh.element_volumes(), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_distorted_mesh_matches_graph_displacement_and_gradient(dimension):
    if dimension == 2:
        mesh = distorted_q4(6, 3, amplitude=0.15, seed=12)
        fixed_nodes = mesh.left_edge_nodes()
        force_node = mesh.node_id(6, 1)
    else:
        mesh = distorted_hex8(5, 3, 3, amplitude=0.15, seed=12)
        fixed_nodes = mesh.x_min_face_nodes()
        force_node = mesh.node_id(5, 1, 2)
    mesh.coords[:, 0] *= 1.3
    fixed = (dimension * fixed_nodes[:, None] + np.arange(dimension)).ravel()
    force = np.zeros((mesh.n_dof, 2))
    force[dimension * force_node + dimension - 1, 0] = -1.0
    force[dimension * force_node, 1] = 0.3
    rho = np.linspace(0.3, 0.9, mesh.n_elements)
    library = ScikitFEMElasticity(mesh.coords, mesh.econn, fixed, force, solver="scipy")
    c, g, u = library.evaluate(rho)
    graph = MechanicsOperator(
        mesh,
        fixed,
        dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
        device="cpu",
        use_fused=True,
        preconditioner="jacobi",
    )
    density = torch.from_numpy(rho)
    state = graph.solve_state(
        density,
        torch.from_numpy(force.T.copy()).reshape(2, -1, dimension),
        rtol=1e-11,
        max_iter=10000,
        allow_direct_fallback=False,
    )
    graph_gradient = graph.compliance_sensitivity(density, state).numpy()
    np.testing.assert_allclose(u, state.u.numpy().reshape(2, -1).T, rtol=1e-8, atol=2e-8)
    assert c == pytest.approx(float(state.compliance), rel=1e-9)
    np.testing.assert_allclose(g, graph_gradient, rtol=1e-8, atol=2e-8)
    direction = np.random.default_rng(7).normal(size=mesh.n_elements) * 0.1
    h = 2e-5
    finite_difference = (
        library.evaluate(rho + h * direction)[0] - library.evaluate(rho - h * direction)[0]
    ) / (2 * h)
    assert g @ direction == pytest.approx(finite_difference, rel=2e-6, abs=1e-6)
    assert library.max_residual < 1e-10


def test_cached_library_form_equals_direct_density_weighted_library_assembly():
    from skfem.models.elasticity import linear_elasticity, plane_stress

    p = make_problem("cantilever", 5, 3)
    library = ScikitFEMElasticity.from_problem(p, solver="scipy")
    rho = np.linspace(0.2, 0.9, p.n_elements)
    modulus = 1e-6 + (1 - 1e-6) * rho**3
    direct = linear_elasticity(*plane_stress(modulus[:, None], 0.3)).assemble(library.basis)
    cached = library.unit_form.fromlocal(library.unit_local * modulus[:, None, None]).tocsr()
    np.testing.assert_allclose(cached.toarray(), direct.toarray(), atol=4e-16, rtol=1e-13)


def test_mature_oc_reduces_compliance_and_enforces_physical_volume():
    p = make_problem("cantilever", 16, 6, volume=0.4, radius=1.5)
    p.coords[:, 0] *= 1.0 + 0.015 * p.coords[:, 0]
    library = ScikitFEMElasticity.from_problem(p, solver="scipy")
    initial = library.evaluate(np.full(p.n_elements, 0.4))[0]
    result = optimize_oc(
        p,
        OCConfig(betas=(1.0,), max_stage_steps=80, min_stage_steps=20, solver="scipy"),
        physics=library,
    )
    assert result["summary"]["C_raw"] < initial * 0.65
    assert result["summary"]["volume"] == pytest.approx(0.4, abs=5e-8)
    c, _, _ = library.evaluate(result["rho"])
    assert c == result["summary"]["C_raw"]


@pytest.mark.parametrize("dimension", [2, 3])
def test_pardiso_symmetric_cholesky_matches_superlu_at_high_contrast(dimension):
    pytest.importorskip("pypardiso")
    mesh = structured_q4(8, 4) if dimension == 2 else structured_hex8(6, 4, 4)
    nodes = mesh.left_edge_nodes() if dimension == 2 else mesh.x_min_face_nodes()
    fixed = (dimension * nodes[:, None] + np.arange(dimension)).ravel()
    force = np.zeros((mesh.n_dof, 2))
    force[-1, 0] = -1.0
    force[-dimension, 1] = 0.3
    rho = np.random.default_rng(8).choice([0.001, 0.3, 0.7, 1.0], size=mesh.n_elements)
    args = mesh.coords, mesh.econn, fixed, force
    library = ScikitFEMElasticity(*args, solver="pypardiso-spd")
    expected = ScikitFEMElasticity(*args, solver="scipy").evaluate(rho)
    for actual, reference in zip(library.evaluate(rho), expected):
        np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=1e-6)
    assert library.last_residual < 1e-8
    library.close()


def test_optional_penalty_continuation_records_actual_model_and_final_p3():
    p = make_problem("cantilever", 8, 4, volume=0.4, radius=1.5)
    result = optimize_oc(
        p,
        OCConfig(
            betas=(1.0, 2.0, 8.0), penalties=(1.0, 2.0, 3.0), max_stage_steps=2, solver="scipy"
        ),
    )
    assert [r["penalty"] for r in result["history"]] == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert result["summary"]["fem_backend"]["penalty"] == 3.0
    independent = ScikitFEMElasticity.from_problem(p, solver="scipy")
    assert independent.evaluate(result["rho"])[0] == result["summary"]["C_raw"]
