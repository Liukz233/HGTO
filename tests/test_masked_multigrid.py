"""A cutout must remain mechanically absent when embedded for preconditioning."""

import numpy as np
import pytest
import torch
from hgto.fem import MechanicsOperator
from hgto.fem.mesh.q4 import structured_q4, Q4Mesh, distorted_q4
from hgto.fem.solvers.masked_mg import CartesianEmbedding
from hgto.reference import ScikitFEMElasticity


@pytest.mark.parametrize("contrast", [False, True])
def test_cartesian_cutout_multigrid_matches_library(contrast):
    original = structured_q4(16, 16)
    center = original.element_centroids()
    cells = original.econn[~((center[:, 0] > 8) & (center[:, 1] > 8))]
    used = np.unique(cells)
    renumber = np.full(original.n_nodes, -1)
    renumber[used] = np.arange(len(used))
    mesh = Q4Mesh(16, 16, original.coords[used], renumber[cells])
    fixed_nodes = np.flatnonzero(mesh.coords[:, 1] == 16)
    fixed = (2 * fixed_nodes[:, None] + np.arange(2)).reshape(-1)
    forces = np.zeros((mesh.n_dof, 1))
    load = np.flatnonzero((mesh.coords[:, 0] == 16) & (mesh.coords[:, 1] == 8))[0]
    forces[2 * load + 1] = -1
    rho = np.linspace(0.2, 0.9, mesh.n_elements)
    if contrast:
        rho[::7] = 0.001
    library = ScikitFEMElasticity(mesh.coords, mesh.econn, fixed, forces, solver="scipy")
    c, g, u = library.evaluate(rho)
    graph = MechanicsOperator(
        mesh,
        fixed,
        dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
        device="cpu",
        use_fused=True,
        preconditioner="mgcg_masked",
        mgcg_coarsest_dtype="float32",
        mgcg_semi_coarsen=True,
    )
    density = torch.from_numpy(rho)
    state = graph.solve_state(
        density,
        torch.from_numpy(forces.T).reshape(1, -1, 2),
        rtol=1e-10,
        max_iter=10000,
        allow_direct_fallback=False,
    )
    assert not bool(state.fallback_used.any())
    assert float(state.residual_rel.max()) <= 1e-10
    assert float(state.compliance) == pytest.approx(c, rel=1e-9)
    np.testing.assert_allclose(state.u.numpy().reshape(-1, 1), u, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(
        graph.compliance_sensitivity(density, state).numpy(), g, rtol=1e-7, atol=1e-7
    )
    assert len(CartesianEmbedding.from_mesh(mesh).elements) == 192
    library.close()


def test_embedding_rejects_non_cartesian_geometry():
    with pytest.raises(ValueError, match="Cartesian"):
        CartesianEmbedding.from_mesh(distorted_q4(8, 8, amplitude=0.1, seed=1))
