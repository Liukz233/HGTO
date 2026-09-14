"""Independent Q4/Hex8 elasticity assembled by scikit-fem.

No HGTO shape-function, quadrature, constitutive or element-stiffness code is
used here.  scikit-fem supplies the mesh mapping, vector basis, quadrature,
elasticity weak form, elemental assembly and sparse scatter.  Linear SIMP
material scaling permits reuse of the library's unit-modulus elemental form.
"""

from importlib.metadata import version
import time

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import triu


class ScikitFEMElasticity:
    """Small-strain isotropic elasticity with interleaved input/output DOFs.

    ``cells`` follow HGTO's Q4 corner ordering (bottom face CCW), or its Hex8
    ordering (bottom face CCW, then the corresponding top face).  ``forces``
    is ``(dimension * n_nodes, n_loads)`` or one vector.  Compliance sums the
    load cases; load weighting must therefore be included in ``forces``.
    ``evaluate`` returns ``(compliance, gradient, displacement)`` with the
    displacement always a two-dimensional array in the original DOF order.
    """

    def __init__(
        self,
        coords,
        cells,
        fixed,
        forces,
        *,
        dimension=None,
        E0=1.0,
        Emin=1e-6,
        nu=0.3,
        penalty=3.0,
        thickness=1.0,
        solver="auto",
        residual_tolerance=1e-7,
    ):
        try:
            from skfem import (
                Basis,
                ElementHex1,
                ElementQuad1,
                ElementVector,
                Functional,
                MeshHex,
                MeshQuad,
            )
            from skfem.models.elasticity import (
                lame_parameters,
                linear_elasticity,
                linear_stress,
                plane_stress,
            )
            from skfem.helpers import ddot, sym_grad
        except ImportError as exc:
            raise ImportError(
                "Install the mature FEM backend: pip install scikit-fem==12.0.2"
            ) from exc

        self.coords = np.asarray(coords, dtype=np.float64)
        self.cells = np.asarray(cells, dtype=np.int64)
        self.dimension = self.coords.shape[1] if dimension is None else dimension
        if self.dimension not in (2, 3) or self.coords.shape[1] != self.dimension:
            raise ValueError("Expected two- or three-dimensional node coordinates")
        if self.cells.ndim != 2 or self.cells.shape[1] != 2**self.dimension:
            raise ValueError("Expected Q4 or Hex8 connectivity")
        if not (0 < Emin <= E0 and -0.99 < nu < 0.499 and penalty >= 1):
            raise ValueError("Invalid isotropic SIMP material parameters")
        if thickness <= 0 or (self.dimension == 3 and thickness != 1.0):
            raise ValueError("Use positive Q4 thickness, or thickness=1 for Hex8")
        self.E0, self.Emin, self.nu = float(E0), float(Emin), float(nu)
        self.penalty, self.thickness = float(penalty), float(thickness)
        self.residual_tolerance = residual_tolerance
        self.n_elements = len(self.cells)
        self.n_dofs = self.dimension * len(self.coords)
        self.fixed = np.unique(np.asarray(fixed, dtype=np.int64))
        if self.fixed.size == 0 or self.fixed.min() < 0 or self.fixed.max() >= self.n_dofs:
            raise ValueError("Fixed DOFs must be valid original interleaved DOF indices")
        self.forces = np.asarray(forces, dtype=np.float64)
        if self.forces.ndim == 1:
            self.forces = self.forces[:, None]
        if self.forces.ndim != 2 or self.forces.shape[0] != self.n_dofs:
            raise ValueError("Forces must have shape (n_dofs, n_loads)")
        self.forces = self.forces.copy()

        if self.dimension == 2:
            # ElementQuad1: (0,0), (1,0), (1,1), (0,1).
            self.corner_permutation = np.arange(4)
            native_mesh = MeshQuad(self.coords.T, self.cells.T, sort_t=False)
            element = ElementVector(ElementQuad1())
            lame = plane_stress(1.0, self.nu)
        else:
            # ElementHex1: 111, 110, 101, 011, 100, 010, 001, 000.
            self.corner_permutation = np.array([6, 2, 5, 7, 1, 3, 4, 0])
            native_mesh = MeshHex(
                self.coords.T, self.cells[:, self.corner_permutation].T, sort_t=False
            )
            element = ElementVector(ElementHex1())
            lame = lame_parameters(1.0, self.nu)
        self.basis = Basis(native_mesh, element, intorder=3)
        # Library-selected order three is the 2^d Gauss rule for Q1 elements.
        if self.basis.X.shape[1] != 2**self.dimension:
            raise RuntimeError("Unexpected scikit-fem quadrature rule")
        determinant = self.basis.mapping.detDF(self.basis.X)
        if np.any(determinant <= 0):
            raise ValueError("Inverted or degenerate element geometry")
        self.element_volumes = self.basis.dx.sum(axis=1) * self.thickness
        self.unit_form = linear_elasticity(*lame).elemental(self.basis)
        self.unit_local = self.unit_form.tolocal() * self.thickness
        stress = linear_stress(*lame)

        @Functional
        def strain_energy(w):
            strain = sym_grad(w.displacement)
            return ddot(stress(strain), strain)

        self._strain_energy = strain_energy
        self.element_dofs = self.basis.element_dofs.T
        # Explicit mapping rather than assuming a particular library numbering.
        self.native_for_input = self.basis.nodal_dofs.T.ravel()
        if len(self.native_for_input) != self.n_dofs:
            raise RuntimeError("The library basis contains unexpected non-nodal DOFs")
        self.native_forces = np.zeros((self.basis.N, self.forces.shape[1]))
        self.native_forces[self.native_for_input] = self.forces
        native_fixed = self.native_for_input[self.fixed]
        self.free = np.setdiff1d(np.arange(self.basis.N), native_fixed)
        self.solver_name = "scipy-superlu"
        self._pardiso = None
        if solver not in ("auto", "scipy", "pypardiso", "pypardiso-spd"):
            raise ValueError("solver must be auto, scipy, pypardiso, or pypardiso-spd")
        if solver in ("auto", "pypardiso", "pypardiso-spd"):
            try:
                from pypardiso import PyPardisoSolver

                self._pardiso = PyPardisoSolver(mtype=2 if solver == "pypardiso-spd" else 11)
                self.solver_name = "pypardiso-spd" if solver == "pypardiso-spd" else "pypardiso"
            except ImportError:
                if solver.startswith("pypardiso"):
                    raise
        self.calls = 0
        self.last_residual = self.max_residual = 0.0
        self.assembly_seconds = self.solve_seconds = 0.0
        self.last_energy_relative_error = 0.0

    @classmethod
    def from_problem(cls, problem, **kwargs):
        """Accept either the Q4 Problem or structured Problem3DSpec API."""
        coords = problem.coords() if callable(problem.coords) else problem.coords
        cells = problem.cells if hasattr(problem, "cells") else problem.econn
        cells = cells() if callable(cells) else cells
        fixed = problem.fixed if hasattr(problem, "fixed") else problem.fixed_dofs
        forces = problem.forces if hasattr(problem, "forces") else problem.force
        return cls(coords, cells, fixed, forces, **kwargs)

    def evaluate(self, rho):
        density = np.asarray(rho, dtype=np.float64).reshape(-1)
        if density.shape != (self.n_elements,) or not np.all(np.isfinite(density)):
            raise ValueError("Expected one finite physical density per element")
        if np.any(density < -2e-14) or np.any(density > 1.0 + 2e-14):
            raise ValueError("Physical densities must lie in [0, 1]")
        # Row-normalized sparse filtering can exceed an endpoint by roundoff.
        density = np.clip(density, 0.0, 1.0)
        started = time.perf_counter()
        modulus = self.Emin + (self.E0 - self.Emin) * density**self.penalty
        matrix = self.unit_form.fromlocal(self.unit_local * modulus[:, None, None]).tocsr()
        reduced = matrix[self.free][:, self.free].tocsr()
        reduced.sort_indices()
        self.assembly_seconds += time.perf_counter() - started
        started = time.perf_counter()
        rhs = self.native_forces[self.free]
        if self._pardiso is not None:
            # Intel PARDISO mtype=2 requires only the upper CSR triangle.
            # The full library matrix remains the residual/energy oracle.
            solve_matrix = (
                triu(reduced, format="csr") if self.solver_name == "pypardiso-spd" else reduced
            )
            solved = self._pardiso.solve(solve_matrix, rhs)
        else:
            solved = splu(reduced.tocsc()).solve(rhs)
        if solved.ndim == 1:
            solved = solved[:, None]
        u = np.zeros_like(self.native_forces)
        u[self.free] = solved
        self.solve_seconds += time.perf_counter() - started
        residual = reduced @ solved - rhs
        self.last_residual = float(
            np.max(
                np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
            )
        )
        self.max_residual = max(self.max_residual, self.last_residual)
        if not np.isfinite(u).all() or self.last_residual > self.residual_tolerance:
            raise RuntimeError(f"Mature FEM equilibrium residual {self.last_residual:.3e}")
        # Integrating strain avoids cancellation in u_e.T K_e u_e when a
        # weakly connected cell undergoes a large nearly rigid motion.
        energy = np.zeros(self.n_elements)
        for load in range(u.shape[1]):
            energy += self.thickness * self._strain_energy.elemental(
                self.basis, displacement=self.basis.interpolate(u[:, load])
            )
        compliance = float(np.sum(self.native_forces * u))
        self.last_energy_relative_error = abs(float(modulus @ energy) - compliance) / max(
            abs(compliance), 1e-30
        )
        gradient = -self.penalty * (self.E0 - self.Emin) * density ** (self.penalty - 1) * energy
        self.calls += 1
        return compliance, gradient, u[self.native_for_input]

    def metadata(self):
        return dict(
            library="scikit-fem",
            library_version=version("scikit-fem"),
            element="ElementQuad1" if self.dimension == 2 else "ElementHex1",
            model="plane stress" if self.dimension == 2 else "3D linear elasticity",
            basis="ElementVector",
            assembly="linear_elasticity.elemental; COOData.fromlocal.tocsr",
            quadrature_points=int(self.basis.X.shape[1]),
            corner_permutation=self.corner_permutation.tolist(),
            solver=self.solver_name,
            linear_solver_version=version("pypardiso" if self._pardiso is not None else "scipy"),
            E0=self.E0,
            Emin=self.Emin,
            nu=self.nu,
            penalty=self.penalty,
            thickness=self.thickness,
            calls=self.calls,
            max_residual=self.max_residual,
            energy_relative_error=self.last_energy_relative_error,
            assembly_seconds=self.assembly_seconds,
            solve_seconds=self.solve_seconds,
        )

    def close(self):
        if self._pardiso is not None:
            self._pardiso.free_memory(everything=True)
