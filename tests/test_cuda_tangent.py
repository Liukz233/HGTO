"""GPU LU must assemble overlapping cells, transpose, and refactor changed values."""

import numpy as np
import pytest
import torch
from hgto.fem import MechanicsOperator
from hgto.fem.mesh.q4 import structured_q4
from hgto.fem.solvers.cudss_tangent import solve_cuda_tangent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_general_tangent_assembly_transpose_and_refactor():
    pytest.importorskip("nvidia.cu12")
    mesh = structured_q4(2, 1)
    op = MechanicsOperator(
        mesh,
        np.array([0, 1], dtype=np.int64),
        dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
        device="cuda:0",
        use_fused=False,
    )
    local = torch.tensor(
        np.random.default_rng(2).normal(size=(2, 8, 8)), device=op.device, dtype=torch.float64
    )
    local += torch.eye(8, device=op.device) * 5
    dofs = (
        2 * torch.as_tensor(mesh.econn, device=op.device)[:, :, None]
        + torch.arange(2, device=op.device)
    ).reshape(2, 8)
    try:
        for transpose in (False, True, False):
            local = local + torch.eye(8, device=op.device) * 0.2
            dense = torch.zeros((op.n_dof, op.n_dof), device=op.device, dtype=torch.float64)
            for e in range(2):
                dense.index_put_((dofs[e, :, None], dofs[e, None, :]), local[e], accumulate=True)
            dense = dense[op.free_dof_mask][:, op.free_dof_mask]
            if transpose:
                dense = dense.T
            rhs = torch.arange(1.0, len(dense) + 1, device=op.device, dtype=torch.float64)
            x, res, _, ok = solve_cuda_tangent(
                op, local, rhs, lambda v: dense @ v, 1e-10, transpose
            )
            assert ok and res < 1e-10 and x.device == op.device
            torch.testing.assert_close(x, torch.linalg.solve(dense, rhs), atol=1e-10, rtol=1e-10)
        assert op._cudss_tangent.solves == 3
    finally:
        if hasattr(op, "_cudss_tangent"):
            op._cudss_tangent.close()
