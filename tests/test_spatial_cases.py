"""Physical boundary-condition checks for the conventional 3D benchmark."""

import numpy as np

from hgto.case_studies import make_3d_case


def test_cantilever_finite_traction_work_and_clamp():
    problem, mesh = make_3d_case("cantilever_3d")
    force = problem.force.reshape(-1, 3)
    loaded = np.flatnonzero(np.linalg.norm(force, axis=1))
    expected_fixed = (3 * np.flatnonzero(mesh.coords[:, 0] == 0)[:, None] + np.arange(3)).ravel()
    assert problem.n_elem == 12288
    assert problem.volfrac == 0.3
    assert np.array_equal(problem.fixed_dofs, expected_fixed)
    assert len(loaded) == 9
    assert np.allclose(force.sum(0), [0.0, -1.0, 0.0])
    assert np.allclose(np.cross(mesh.coords - [0.0, 8.0, 8.0], force).sum(0), [0.0, 0.0, -48.0])
    # A bilinear displacement field has an exactly integrable mean over the
    # finite patch. These values detect equal-node weighting instead of the
    # intended consistent Q4 face quadrature.
    x, y, z = mesh.coords.T
    uy = 1.3 + 0.7 * y - 0.4 * z + 0.12 * y * z
    expected_work = -(1.3 + 0.7 * 8.0 - 0.4 * 8.0 + 0.12 * 8.0 * 8.0)
    assert np.isclose(force[:, 1] @ uy, expected_work, rtol=0.0, atol=1e-12)
    assert not problem.extra["prescribed_solid_elements"]
    assert not problem.extra["symmetry_constraints"]
