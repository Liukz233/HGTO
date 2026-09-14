"""Unstructured Q4 SA-AMG preconditioning, with device-resident V cycles.

CPU PyAMG constructs the hierarchy; all applications use torch on the state
 device. Only the preconditioner is assembled. The graph stiffness action and
 true residual still use the current density. Setup is part of run timing.
"""

import numpy as np
import torch
from .linear import assemble_sparse_stiffness


def _tensor(matrix, device, dtype):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(matrix.indptr, device=device, dtype=torch.int64),
        torch.as_tensor(matrix.indices, device=device, dtype=torch.int64),
        torch.as_tensor(matrix.data, device=device, dtype=dtype),
        size=matrix.shape,
        device=device,
        dtype=dtype,
    )


def _mv(matrix, vector):
    return torch.mv(matrix, vector)


class AlgebraicMGHierarchy:
    def __init__(self, operator, youngs):
        from pyamg import smoothed_aggregation_solver

        free = operator.free_dof_index.detach().cpu().numpy()
        stiffness = assemble_sparse_stiffness(operator.Ke0, youngs, operator.econn, operator.n_dof)
        reduced = stiffness[free][:, free].tocsr()
        xy = operator.coords.detach().cpu().numpy()
        modes = np.zeros((operator.n_dof, 3))
        modes[0::2, 0] = 1.0
        modes[1::2, 1] = 1.0
        modes[0::2, 2] = -(xy[:, 1] - xy[:, 1].mean())
        modes[1::2, 2] = xy[:, 0] - xy[:, 0].mean()
        # Explicit random state avoids changing the network RNG sequence.
        rng_state = np.random.get_state()
        try:
            np.random.seed(0)
            hierarchy = smoothed_aggregation_solver(
                reduced,
                B=modes[free],
                symmetry="symmetric",
                max_coarse=64,
                improve_candidates=None,
                coarse_solver="cholesky",
            )
        finally:
            np.random.set_state(rng_state)
        self.free = operator.free_dof_index
        self.n_dof = operator.n_dof
        self.levels = []
        for level in hierarchy.levels[:-1]:
            matrix = level.A.tocsr()
            diagonal = matrix.diagonal()
            if np.any(diagonal <= 0):
                raise ValueError("AMG requires positive diagonal")
            # Gershgorin bound gives a conservative SPD Jacobi smoother.
            bound = np.max(np.asarray(abs(matrix).sum(axis=1)).ravel() / diagonal)
            self.levels.append(
                dict(
                    A=_tensor(matrix, operator.device, operator.dtype),
                    P=_tensor(level.P, operator.device, operator.dtype),
                    R=_tensor(level.P.T, operator.device, operator.dtype),
                    inverse_diagonal=torch.as_tensor(
                        0.9 / bound / diagonal, device=operator.device, dtype=operator.dtype
                    ),
                )
            )
        coarse = hierarchy.levels[-1].A.toarray()
        coarse = 0.5 * (coarse + coarse.T)
        self.cholesky = torch.linalg.cholesky(
            torch.as_tensor(coarse, device=operator.device, dtype=operator.dtype)
        )
        self.coarsest_fp32_fallback = False
        self.sizes = [level.A.shape[0] for level in hierarchy.levels]

    def _cycle(self, index, rhs):
        if index == len(self.levels):
            return torch.cholesky_solve(rhs[:, None], self.cholesky).reshape(-1)
        level = self.levels[index]
        x = torch.zeros_like(rhs)
        for _ in range(3):
            x = x + level["inverse_diagonal"] * (rhs - _mv(level["A"], x))
        correction = self._cycle(index + 1, _mv(level["R"], rhs - _mv(level["A"], x)))
        x = x + _mv(level["P"], correction)
        for _ in range(3):
            x = x + level["inverse_diagonal"] * (rhs - _mv(level["A"], x))
        return x

    def apply(self, rhs):
        result = torch.zeros_like(rhs)
        result[self.free] = self._cycle(0, rhs[self.free])
        return result

    def stamp(self):
        return dict(
            preconditioner="smoothed aggregation AMG",
            levels=len(self.sizes),
            level_dofs=self.sizes,
            setup_device="cpu",
            apply_device=str(self.cholesky.device),
            coarsest_dtype="float64",
        )
