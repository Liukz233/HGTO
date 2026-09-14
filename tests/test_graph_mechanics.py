"""Compare graph mechanics with an independently assembled Q4 reference."""

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from hgto.fem import MechanicsOperator
from hgto.fem.mesh import structured_q4, structured_hex8


def independent_q4_stiffness(coords, poisson):
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    constitutive = np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1 - poisson) / 2]]
    ) / (1 - poisson**2)
    stiffness = np.zeros((8, 8))
    for xi, eta in corners / np.sqrt(3):
        gradients = (
            np.column_stack(
                (
                    corners[:, 0] * (1 + eta * corners[:, 1]),
                    corners[:, 1] * (1 + xi * corners[:, 0]),
                )
            )
            / 4
        )
        jacobian = coords.T @ gradients
        physical = gradients @ np.linalg.inv(jacobian)
        strain = np.zeros((3, 8))
        strain[0, ::2] = physical[:, 0]
        strain[1, 1::2] = physical[:, 1]
        strain[2, ::2] = physical[:, 1]
        strain[2, 1::2] = physical[:, 0]
        stiffness += strain.T @ constitutive @ strain * np.linalg.det(jacobian)
    return stiffness


@pytest.mark.parametrize("preconditioner", ["jacobi", "mgcg"])
def test_multiload_solution_and_sensitivity(preconditioner):
    mesh = structured_q4(12, 4)
    fixed = np.sort(np.ravel([2 * mesh.left_edge_nodes(), 2 * mesh.left_edge_nodes() + 1]))
    material = dict(E0=1.0, Emin=1.0e-6, nu=0.3, p=3.0)
    operator = MechanicsOperator(
        mesh,
        fixed,
        material,
        use_fused=True,
        preconditioner=preconditioner,
        mgcg_coarsest_dtype="float32",
        mgcg_semi_coarsen=True,
    )
    force = torch.zeros((2, mesh.n_nodes, 2))
    force[0, mesh.node_id(12, 2), 1] = -1.0
    force[1, mesh.node_id(12, 4), 0] = 0.4
    density = torch.linspace(0.35, 0.8, mesh.n_elements)
    state = operator.solve_state(
        density, force, rtol=1e-10, max_iter=10000, allow_direct_fallback=False
    )

    # This assembly uses its own integration, connectivity expansion and solve.
    local = np.array([independent_q4_stiffness(mesh.coords[c], 0.3) for c in mesh.econn])
    dofs = (2 * mesh.econn[..., None] + np.arange(2)).reshape(mesh.n_elements, 8)
    moduli = 1e-6 + (1 - 1e-6) * density.numpy() ** 3
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    matrix = sparse.coo_matrix(
        ((local * moduli[:, None, None]).ravel(), (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)
    ).tocsc()
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    rhs = force.numpy().reshape(2, -1).T
    displacement = np.zeros_like(rhs)
    displacement[free] = spsolve(matrix[free][:, free], rhs[free])
    np.testing.assert_allclose(state.u.numpy().reshape(2, -1).T, displacement, rtol=1e-8, atol=1e-8)
    assert float(state.residual_rel.max()) < 1e-10
    assert float(state.compliance) == pytest.approx(np.sum(rhs * displacement), rel=1e-9)

    direction = torch.linspace(-0.15, 0.17, mesh.n_elements)
    gradient = operator.compliance_sensitivity(density, state)
    step = 1e-5
    samples = [
        operator.solve_state(
            density + sign * step * direction,
            force,
            rtol=1e-11,
            max_iter=10000,
            allow_direct_fallback=False,
        ).compliance
        for sign in [1, -1]
    ]
    finite_difference = (samples[0] - samples[1]) / (2 * step)
    assert float(gradient @ direction) == pytest.approx(float(finite_difference), rel=2e-5)


def test_hex8_rigid_translation_and_fused_operator():
    mesh = structured_hex8(4, 2, 2)
    fixed_nodes = mesh.x_min_face_nodes()
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
    operator = MechanicsOperator(mesh, fixed, dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0))
    density = torch.linspace(0.3, 0.9, mesh.n_elements)
    torch.manual_seed(3)
    direction = torch.randn((1, mesh.n_nodes, 3))
    chain = operator.jvp_u(direction, density)
    operator.use_fused = True
    fused = operator.jvp_u(direction, density)
    torch.testing.assert_close(chain, fused, rtol=1e-12, atol=1e-12)
    assert float((direction * fused).sum()) > 0

    # A rigid translation has zero strain before boundary restriction.
    translation = torch.ones((1, mesh.n_nodes, 3))
    torch.testing.assert_close(
        operator._chain_matvec(translation, density),
        torch.zeros_like(translation),
        atol=1e-14,
        rtol=0,
    )


def test_multilevel_preconditioner_reaches_same_equilibrium():
    mesh = structured_q4(32, 16)
    fixed = (2 * mesh.left_edge_nodes()[:, None] + np.arange(2)).ravel()
    operator = MechanicsOperator(
        mesh,
        fixed,
        dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
        use_fused=True,
        preconditioner="mgcg",
        mgcg_coarsest_dtype="float32",
        mgcg_semi_coarsen=True,
    )
    force = torch.zeros((1, mesh.n_nodes, 2))
    force[0, mesh.node_id(32, 8), 1] = -1.0
    density = torch.linspace(0.4, 0.8, mesh.n_elements)
    coarse = operator.solve_state(
        density, force, rtol=1e-9, max_iter=10000, allow_direct_fallback=False
    )
    operator.preconditioner = "jacobi"
    reference = operator.solve_state(
        density, force, rtol=1e-9, max_iter=10000, allow_direct_fallback=False
    )
    torch.testing.assert_close(coarse.u, reference.u, atol=1e-7, rtol=1e-8)
    assert float(coarse.residual_rel.max()) < 1e-9
