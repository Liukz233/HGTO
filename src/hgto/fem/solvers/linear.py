"""Linear solvers with explicit checks of the free-DOF residual.

The convergence value is the recomputed norm of A x - b divided by the norm
of b. A SolveFailure contains the achieved residual when convergence fails."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

Matvec = Callable[[torch.Tensor], torch.Tensor]


class SolveFailure(RuntimeError):
    """A linear/nonlinear solve failed to certify convergence."""

    def __init__(self, message: str, *, kind: str, residual_rel: float, iterations: int) -> None:
        super().__init__(message)
        self.kind = kind  # "pcg" | "direct" | "minres" | "newton"
        self.residual_rel = residual_rel
        self.iterations = iterations


def pcg_callback(
    matvec: Matvec,
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    initial: torch.Tensor,
    rtol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Preconditioned conjugate gradients with a Jacobi diagonal.

    Convergence is checked on the recomputed residual of the reduced system.
    Scalar convergence checks share a host transfer on GPU. Returns the
    solution, relative residual, iteration count and convergence indicator."""
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    diagonal_ok = torch.all(torch.isfinite(diagonal) & (diagonal > 0.0))
    if not bool(diagonal_ok.item()):
        raise SolveFailure(
            "Jacobi diagonal must be finite and positive",
            kind="pcg",
            residual_rel=float("nan"),
            iterations=0,
        )

    x = initial.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    normalizer = torch.where(rhs_norm > 0.0, rhs_norm, torch.ones_like(rhs_norm))
    residual = rhs - matvec(x)
    residual_rel = torch.linalg.vector_norm(residual) / normalizer
    if float(residual_rel.item()) <= rtol:
        return x, residual_rel, 0, True

    z = residual / diagonal
    direction = z.clone()
    rz = torch.dot(residual, z)
    rz_value: float | None = None  # read lazily with the first fused copy
    converged = False
    iterations = 0
    pending_rz: torch.Tensor | None = None  # rz_new whose check is deferred
    for iteration in range(1, max_iter + 1):
        product = matvec(direction)
        curvature = torch.dot(direction, product)
        alpha = rz / curvature
        x_candidate = x + alpha * direction
        residual_candidate = rhs - matvec(x_candidate)  # true residual (certificate)
        residual_rel_candidate = torch.linalg.vector_norm(residual_candidate) / normalizer
        if pending_rz is None:
            rz_value, curvature_value, residual_rel_value = torch.stack(
                [rz, curvature, residual_rel_candidate]
            ).tolist()
        else:
            pending_value, curvature_value, residual_rel_value = torch.stack(
                [pending_rz, curvature, residual_rel_candidate]
            ).tolist()
            # the historical post-update check of the previous iteration
            if not math.isfinite(pending_value) or rz_value == 0.0:
                break
            rz_value = pending_value
        if not math.isfinite(curvature_value) or curvature_value <= 0.0:
            break  # indefinite/broken operator: certify failure below
        x = x_candidate
        residual = residual_candidate
        residual_rel = residual_rel_candidate
        iterations = iteration
        if residual_rel_value <= rtol:
            converged = True
            break
        if iteration == max_iter:
            break  # the direction update would be dead work
        z = residual / diagonal
        rz_new = torch.dot(residual, z)
        beta = rz_new / rz
        direction = z + beta * direction
        pending_rz = rz_new
        rz = rz_new

    return x, residual_rel, iterations, converged


def assemble_sparse_stiffness(
    Ke0: torch.Tensor,
    E_e: torch.Tensor,
    econn: torch.Tensor,
    n_dof: int,
) -> sparse.csr_matrix:
    """Assemble the global CSR stiffness from pre-integrated unit element matrices.

    Dimension-generic: the interleave stride is derived from the element
    matrix width (Q4: 8 local dofs / stride 2; hex8: 24 local dofs / stride 3).
    """
    Ke = (Ke0 * E_e[:, None, None]).cpu().numpy()
    econn_np = econn.cpu().numpy()
    n_elements, nodes_per_element = econn_np.shape
    n_local = Ke.shape[1]
    n_dim = n_local // nodes_per_element
    edofs = np.empty((n_elements, n_local), dtype=np.int64)
    for axis in range(n_dim):
        edofs[:, axis::n_dim] = n_dim * econn_np + axis
    rows = np.repeat(edofs, n_local, axis=1).ravel()
    cols = np.tile(edofs, (1, n_local)).ravel()
    K = sparse.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


def direct_solve_reduced(
    K: sparse.csr_matrix,
    rhs: torch.Tensor,
    free_mask: torch.Tensor,
    rtol: float,
    max_refinement: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Certified sparse direct solve of the free-DOF system.

    High-contrast SIMP fields (condition ~1e6+) limit a single LU backsolve
    to ~1e-8 relative residual; iterative refinement (reusing the same
    factorization) recovers the lost digits before the certificate is issued.
    Returns the FULL-space solution field vector (fixed DOFs zero), plus the
    true-residual certificate computed against the reduced system.
    """
    free = free_mask.cpu().numpy()
    K_ff = K[free][:, free].tocsc()
    rhs_np = rhs.detach().cpu().numpy()
    rhs_f = rhs_np[free]
    lu = sparse_linalg.splu(K_ff)
    x_f = lu.solve(rhs_f)
    norm = np.linalg.norm(rhs_f)
    normalizer = norm if norm > 0.0 else 1.0
    residual = rhs_f - K_ff @ x_f
    residual_rel = float(np.linalg.norm(residual) / normalizer)
    for _ in range(max_refinement):
        if residual_rel <= rtol:
            break
        correction = lu.solve(residual)
        candidate = x_f + correction
        candidate_residual = rhs_f - K_ff @ candidate
        candidate_rel = float(np.linalg.norm(candidate_residual) / normalizer)
        if not np.isfinite(candidate_rel) or candidate_rel >= residual_rel:
            break  # refinement stagnated; certify honestly with the best iterate
        x_f, residual, residual_rel = candidate, candidate_residual, candidate_rel
    x_full = np.zeros_like(rhs_np)
    x_full[free] = x_f
    x = torch.as_tensor(x_full, dtype=rhs.dtype, device=rhs.device)
    certificate = rhs.new_tensor(residual_rel)
    return x, certificate, residual_rel <= rtol
