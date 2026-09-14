"""Check optional fast tangent solves against original equations and SciPy."""

from pathlib import Path
import numpy as np
import pytest
import torch
from hgto.nonlinear.optimization import case, operator, solve
from hgto.fem.solvers.tangent import solve_sparse_tangent
from hgto.fem.mesh.q4 import structured_q4
from hgto.fem import MechanicsOperator


@pytest.mark.parametrize("transpose", [False, True])
def test_pardiso_general_indefinite_tangent(transpose):
    pytest.importorskip("pypardiso")
    torch.set_num_threads(2)
    mesh = structured_q4(1, 1)
    op = MechanicsOperator(
        mesh,
        np.array([], dtype=np.int64),
        dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
        device="cpu",
        use_fused=False,
    )
    op.sparse_tangent_backend = "pypardiso"
    rng = np.random.default_rng(2)
    a = rng.normal(size=(8, 8)) + 0.5 * np.eye(8)
    # Assembly follows the element's local DOF ordering, not global order.
    ids = (2 * mesh.econn[0, :, None] + np.arange(2)).reshape(-1)
    global_a = np.zeros((8, 8))
    global_a[np.ix_(ids, ids)] = a
    target = torch.tensor(global_a.T.copy() if transpose else global_a, dtype=torch.float64)
    b = torch.arange(1.0, 9.0, dtype=torch.float64)
    x, res, _, ok = solve_sparse_tangent(
        op,
        torch.tensor(a[None], dtype=torch.float64),
        b,
        lambda v: target @ v,
        1e-10,
        transpose=transpose,
    )
    assert ok and res < 1e-10
    torch.testing.assert_close(x, torch.linalg.solve(target, b), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("penalty", [1.0, 3.0])
def test_full_bridge_state_and_sensitivity_match_scipy(penalty):
    pytest.importorskip("pypardiso")
    torch.set_num_threads(2)
    with torch.device("cpu"):
        setup, _ = case("bridge_nh")
        force = torch.tensor(setup.f, dtype=torch.float64) * 0.1
        path = Path(__file__).resolve().parent / "fixtures/bridge_density.npy"
        rho = torch.tensor(
            np.load(path) if penalty == 3 else np.full(3072, 0.4), dtype=torch.float64
        )
        values = []
        for backend in ["scipy", "pypardiso"]:
            op = operator(setup, p=penalty, tangent_backend=backend)
            c, g, state = solve(op, rho, force, "nh", True, 12, objective="complementary_work")
            assert state["residual"] <= 1e-8
            values.append((c, g, state))
        a, b = values
        assert a[0] == pytest.approx(b[0], rel=1e-9)
        torch.testing.assert_close(a[1], b[1], rtol=1e-7, atol=1e-10)
        torch.testing.assert_close(a[2]["history_u"], b[2]["history_u"], rtol=1e-7, atol=1e-9)
