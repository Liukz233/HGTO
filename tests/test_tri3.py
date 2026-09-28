"""Linear triangles, passive elements and the beam with reinforced openings."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from hgto.domains.io import load_case
from hgto.fem.mesh.dual_graph import build_element_dual_graph
from hgto.fem.mesh.tri3 import Tri3Mesh, triangle_unit_stiffness
from hgto.linear2d.design import volume_density
from hgto.reference import ScikitFEMElasticity

ROOT = Path(__file__).resolve().parents[1]


def small_tri_mesh(nx=6, ny=3):
    x, y = np.meshgrid(np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1))
    rng = np.random.default_rng(5)
    interior = (x > 0) & (x < 2) & (y > 0) & (y < 1)
    x = x + 0.04 * rng.uniform(-1, 1, x.shape) * interior
    y = y + 0.04 * rng.uniform(-1, 1, y.shape) * interior
    coords = np.c_[x.ravel(), y.ravel()]
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1, n2, n3 = n0 + 1, n0 + nx + 2, n0 + nx + 1
            cells += [[n0, n1, n2], [n0, n2, n3]]
    mesh = Tri3Mesh(0, 0, coords, np.asarray(cells, dtype=np.int64))
    left = np.flatnonzero(np.isclose(coords[:, 0], 0))
    fixed = np.sort(np.r_[2 * left, 2 * left + 1])
    force = np.zeros(mesh.n_dof)
    force[2 * np.flatnonzero(np.isclose(coords[:, 0], 2)) + 1] = -0.25
    return mesh, fixed, force


def assemble(mesh, rho, p=3.0, Emin=1e-6):
    ke = triangle_unit_stiffness(mesh) * (Emin + (1 - Emin) * rho**p)[:, None, None]
    dofs = (2 * mesh.econn[:, :, None] + np.arange(2)).reshape(-1, 6)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    return sparse.coo_matrix((ke.ravel(), (rows, cols)), shape=(mesh.n_dof,) * 2).tocsr()


def test_tri3_stiffness_is_symmetric_with_three_rigid_modes():
    mesh, _, _ = small_tri_mesh()
    ke = triangle_unit_stiffness(mesh)
    np.testing.assert_allclose(ke, ke.transpose(0, 2, 1), atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(ke)
    assert np.all(np.sum(np.abs(eigenvalues) < 1e-12, axis=1) == 3)
    assert np.all(eigenvalues[:, 3:] > 0)
    K = assemble(mesh, np.random.default_rng(1).uniform(0.1, 1, mesh.n_elements))
    assert abs(K - K.T).max() < 1e-13
    flipped = Tri3Mesh(0, 0, mesh.coords, mesh.econn[:, ::-1].copy())
    with pytest.raises(ValueError, match="positive oriented"):
        triangle_unit_stiffness(flipped)


def test_tri3_compliance_and_gradient_match_scikit_fem():
    mesh, fixed, force = small_tri_mesh()
    rho = np.random.default_rng(2).uniform(0.2, 1, mesh.n_elements)
    K = assemble(mesh, rho)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    u = np.zeros(mesh.n_dof)
    u[free] = sparse.linalg.spsolve(K[free][:, free].tocsc(), force[free])
    ue = u[(2 * mesh.econn[:, :, None] + np.arange(2)).reshape(-1, 6)]
    energy = np.einsum("ei,eij,ej->e", ue, triangle_unit_stiffness(mesh), ue)
    reference = ScikitFEMElasticity(
        mesh.coords, mesh.econn, fixed, force[:, None], nu=0.3, solver="scipy"
    )
    compliance, gradient, _ = reference.evaluate(rho)
    reference.close()
    assert compliance == pytest.approx(force @ u, rel=1e-10)
    np.testing.assert_allclose(gradient, -3 * (1 - 1e-6) * rho**2 * energy, rtol=1e-8)
    np.testing.assert_allclose(reference.element_volumes, mesh.element_volumes(), rtol=1e-12)


def test_tri3_dual_graph_connects_edge_neighbours():
    mesh, _, _ = small_tri_mesh(2, 1)
    graph = build_element_dual_graph(mesh)
    pairs = [tuple(pair) for pair in graph["edge_index"].T]
    assert pairs == [(0, 1), (0, 3), (1, 0), (2, 3), (3, 0), (3, 2)]
    shared = {(0, 1): (0, 4), (0, 3): (1, 4), (2, 3): (1, 5)}
    for (a, b), length in zip(pairs, graph["edge_attr"][:, 3]):
        nodes = shared[min(a, b), max(a, b)]
        assert length == pytest.approx(np.linalg.norm(np.subtract(*mesh.coords[list(nodes)])))


def test_passive_elements_keep_density_and_count_in_volume():
    z = torch.linspace(-2, 2, 6, dtype=torch.float64, requires_grad=True)
    fixed = torch.tensor([1.0, float("nan"), float("nan"), 0.0, float("nan"), float("nan")])
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 0.5, 1.5], dtype=torch.float64)
    rho = volume_density(z, 0.45, None, 4.0, 0.001, weights, fixed_density=fixed)
    assert float(rho[0].detach()) == 1.0 and float(rho[3].detach()) == 0.0
    assert float(rho @ weights / weights.sum()) == pytest.approx(0.45, abs=1e-12)
    gradient = torch.autograd.grad(rho @ weights, z)[0]
    assert float(gradient.abs().max()) < 1e-12
    with pytest.raises(ValueError):
        volume_density(z, 0.45, None, 4.0, 0.001, weights, fixed_density=fixed * 0.5)


def test_ring_beam_mesh_loads_as_tri3_with_passive_rings():
    metadata, arrays, mesh = load_case(ROOT / "meshes" / "ring_beam")
    assert isinstance(mesh, Tri3Mesh)
    assert (mesh.n_elements, mesh.n_nodes) == (17043, 8751)
    solid = np.isfinite(arrays["fixed_density"])
    assert solid.sum() == 1868 and np.all(arrays["fixed_density"][solid] == 1)
    np.testing.assert_allclose(mesh.element_volumes(), arrays["element_volumes"], rtol=1e-12)
    np.testing.assert_allclose(
        [arrays["force"][0::2].sum(), arrays["force"][1::2].sum()], [0, -1], atol=1e-14
    )
    assert metadata["target_volume"] == 0.4 and metadata["filter_radius"] == 0.1
    assert np.all(triangle_unit_stiffness(mesh)[:, 0, 0] > 0)


def test_linear_runner_keeps_rings_solid_on_tri3(tmp_path):
    from hgto.linear import run

    settings = {
        "category": "domains",
        "method": "hgto",
        "case": {"mesh": "meshes/ring_beam"},
        "hgto": {
            "n_freq": 8,
            "hidden_dim": 8,
            "stage_steps": [1, 1],
            "betas": [1.0, 8.0],
            "penalties": [1.0, 3.0],
            "device": "cpu",
            "max_updates": 2,
        },
    }
    result = run(settings, tmp_path / "hgto", ROOT)
    rho = result["rho"]
    _, arrays, _ = load_case(ROOT / "meshes" / "ring_beam")
    solid = np.isfinite(arrays["fixed_density"])
    assert np.all(rho[solid] == 1.0)
    assert result["summary"]["volume"] == pytest.approx(0.4, abs=1e-7)
    assert result["summary"]["independent_relative_C_error"] < 1e-8
