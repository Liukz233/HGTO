"""Independent assembled Hex8 states and SIMP sensitivities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from hgto.fem.solvers.sparse import solve_spd, solver_name

from hgto.linear3d.element import element_stiffness

E0, EMIN, NU, PENAL = 1.0, 1e-6, 0.3, 3.0
from hgto.linear3d.problems import Problem3DSpec


class Hex8Backend:
    name = "numpy_hex8(" + solver_name() + ")"

    def __init__(self, problem: Problem3DSpec):
        self.problem = problem
        coords = problem.coords()
        econn = problem.econn()
        # One unit-modulus 24x24 element stiffness, integrated by the oracle.
        self.Ke0 = element_stiffness(coords[econn[0]], 1.0, NU)
        # Interleaved dof table (Ne, 24) and fixed COO triplet layout.
        self.edofs = np.empty((problem.n_elem, 24), dtype=np.int64)
        self.edofs[:, 0::3] = 3 * econn
        self.edofs[:, 1::3] = 3 * econn + 1
        self.edofs[:, 2::3] = 3 * econn + 2
        self._rows = np.repeat(self.edofs, 24, axis=1).reshape(-1)
        self._cols = np.tile(self.edofs, (1, 24)).reshape(-1)
        self._ke_flat = self.Ke0.reshape(-1)
        free_mask = np.ones(problem.n_dof, dtype=bool)
        free_mask[problem.fixed_dofs] = False
        self._free = np.flatnonzero(free_mask)
        self._n_solves = 0

    # ------------------------------------------------------------------ FEM
    # `p` defaults to the frozen SIMP exponent; the nested neural arm passes
    # its penalty-continuation value per step (mirrors PythonQ4Backend).
    def _youngs(self, rho: np.ndarray, p: float = PENAL) -> np.ndarray:
        return EMIN + np.asarray(rho, dtype=np.float64) ** p * (E0 - EMIN)

    def assemble(self, rho: np.ndarray, p: float = PENAL) -> sparse.csr_matrix:
        E_e = self._youngs(rho, p)
        values = np.multiply.outer(E_e, self._ke_flat).reshape(-1)
        return sparse.coo_matrix(
            (values, (self._rows, self._cols)),
            shape=(self.problem.n_dof, self.problem.n_dof),
        ).tocsr()

    def solve(self, rho: np.ndarray, p: float = PENAL) -> Tuple[float, np.ndarray]:
        K = self.assemble(rho, p)
        f = self.problem.force
        u = np.zeros(self.problem.n_dof, dtype=np.float64)
        free = self._free
        u[free] = solve_spd(K[free][:, free].tocsr(), f[free])
        if not np.all(np.isfinite(u)):
            raise RuntimeError("FEM solve returned non-finite displacements")
        self._n_solves += 1
        return float(f @ u), u

    def sensitivity(self, rho: np.ndarray, u: np.ndarray, p: float = PENAL) -> np.ndarray:
        ue = u[self.edofs]  # (Ne, 24)
        ueKue = np.einsum("ei,ij,ej->e", ue, self.Ke0, ue)  # unit strain energy
        rho_arr = np.asarray(rho, dtype=np.float64)
        return -p * rho_arr ** (p - 1.0) * (E0 - EMIN) * ueKue

    def solve_count(self) -> int:
        return self._n_solves
