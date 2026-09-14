"""Explicit sparse tangent-solver callback for CPU nonlinear mechanics.

Pass ``linear_solver=solve_sparse_tangent`` to the nonlinear state and
sensitivity functions. Their default matrix-free solvers remain available.
The callback returns (solution, relative residual, linear solves, converged).
"""

from __future__ import annotations

import weakref
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve


def assemble_element_matrices(operator, element_matrices: torch.Tensor):
    """Assemble Q4 element tangents and restrict them to unconstrained DOFs."""
    expected = (operator.n_elements, 8, 8)
    if tuple(element_matrices.shape) != expected:
        raise ValueError(f"Expected element matrices of shape {expected}")
    if not hasattr(operator, "_sparse_tangent_indices"):
        nodes = operator.econn.detach().cpu().numpy()
        dofs = np.stack((2 * nodes, 2 * nodes + 1), axis=-1).reshape(-1, 8)
        rows = np.repeat(dofs, 8, axis=1).reshape(-1)
        cols = np.tile(dofs, (1, 8)).reshape(-1)
        free = np.flatnonzero(operator.free_dof_mask.detach().cpu().numpy())
        operator._sparse_tangent_indices = (rows, cols, free)
    rows, cols, free = operator._sparse_tangent_indices
    matrix = sparse.coo_matrix(
        (element_matrices.detach().cpu().numpy().reshape(-1), (rows, cols)),
        shape=(operator.n_dof, operator.n_dof),
    ).tocsc()
    return matrix[free][:, free]


def solve_sparse_tangent(operator, element_matrices, rhs, matvec, rtol, transpose=False):
    """Factorize the supplied tangent and certify it against its original action.

    The explicit ``transpose`` flag is used by the NH adjoint. The callback
    never extracts state from Python closures or changes module globals.
    Residual acceptance is max(10*rtol, 1e-10); the outer nonlinear solver
    independently checks equilibrium using its requested tolerance.
    """
    if rhs.device.type != "cpu":
        raise ValueError("The SciPy tangent callback requires CPU state tensors")
    matrix = assemble_element_matrices(operator, element_matrices)
    if transpose:
        matrix = matrix.T.tocsc()
    backend = getattr(operator, "sparse_tangent_backend", "scipy")
    if backend == "scipy":
        solution = spsolve(matrix, rhs.detach().numpy())
    elif backend == "pypardiso":
        if not hasattr(operator, "_pardiso_tangent_solver"):
            from pypardiso import PyPardisoSolver

            solver = PyPardisoSolver(mtype=11)
            operator._pardiso_tangent_solver = solver
            weakref.finalize(operator, solver.free_memory, everything=True)
        # General real matrix type: no assumption of positive definiteness.
        # Each changed tangent is numerically refactorized; only the handle
        # is retained. Certification still uses the original element action.
        solution = operator._pardiso_tangent_solver.solve(matrix.tocsr(), rhs.detach().numpy())
    else:
        raise ValueError(f"Unknown sparse tangent backend: {backend}")
    value = torch.as_tensor(solution, dtype=rhs.dtype, device=rhs.device)
    residual = torch.linalg.vector_norm(matvec(value) - rhs) / torch.linalg.vector_norm(
        rhs
    ).clamp_min(1e-30)
    converged = bool(torch.isfinite(value).all() and residual <= max(10 * float(rtol), 1e-10))
    return value, float(residual), 1, converged
