"""Hex8 meshes with x-fastest, then y, then z node and element ordering.

node_id = iz * (nelx + 1) * (nely + 1) + iy * (nelx + 1) + ix.
Each node has three interleaved displacement components. Local corner order
matches the reference quadrature defined in this package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hgto.fem.mesh.quadrature import hex8_reference_tables


def _structured_econn(nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Binding hex8 connectivity of the complete structured grid."""
    nxp = nelx + 1
    layer = nxp * (nely + 1)
    ez, ey, ex = np.meshgrid(np.arange(nelz), np.arange(nely), np.arange(nelx), indexing="ij")
    n0 = ez.ravel() * layer + ey.ravel() * nxp + ex.ravel()  # (ex, ey, ez)
    bottom = np.stack([n0, n0 + 1, n0 + nxp + 1, n0 + nxp], axis=1)
    return np.concatenate([bottom, bottom + layer], axis=1).astype(np.int64)


@dataclass
class Hex8Mesh:
    nelx: int
    nely: int
    nelz: int
    coords: np.ndarray  # (Nn, 3) float64 node coordinates
    econn: np.ndarray  # (Ne, 8) int64, bottom face CCW then top face CCW
    thickness: float = 1.0  # no 3D thickness concept; kept 1.0 for API parity

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self.econn.shape[0]

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def is_regular(self) -> bool:
        """Whether connectivity is the complete binding structured grid.

        Coordinates may be graded or smoothly mapped: this is deliberately a
        grid-completeness test, not a uniform-spacing test, exactly
        mirroring the Q4 logic in 3D.
        """
        if self.nelx <= 0 or self.nely <= 0 or self.nelz <= 0:
            return False
        if self.n_nodes != (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1):
            return False
        if self.n_elements != self.nelx * self.nely * self.nelz:
            return False
        expected = _structured_econn(self.nelx, self.nely, self.nelz)
        return bool(np.array_equal(self.econn, expected))

    def node_id(self, ix: int, iy: int, iz: int) -> int:
        assert 0 <= ix <= self.nelx and 0 <= iy <= self.nely and 0 <= iz <= self.nelz
        return iz * (self.nelx + 1) * (self.nely + 1) + iy * (self.nelx + 1) + ix

    def element_id(self, ex: int, ey: int, ez: int) -> int:
        assert 0 <= ex < self.nelx and 0 <= ey < self.nely and 0 <= ez < self.nelz
        return ez * self.nelx * self.nely + ey * self.nelx + ex

    def element_centroids(self) -> np.ndarray:
        return self.coords[self.econn].mean(axis=1)

    def element_volumes(self) -> np.ndarray:
        """Gauss-summed Jacobian volumes (BINDING): sum_g w_g * detJ_g.

        Exactly 1.0 on unit cubes. Exact for any trilinear hex: detJ has
        polynomial degree <= 2 per reference coordinate and the 2x2x2 rule
        integrates through degree 3 per coordinate.
        """
        tables = hex8_reference_tables()
        x = self.coords[self.econn]  # (Ne, 8, 3)
        jac = np.einsum("eai,gaj->egij", x, tables["dN_dxi"])
        return np.linalg.det(jac) @ tables["weights"]

    # ---- boundary selectors (node ids, ascending) ----
    def _node_grid(self) -> np.ndarray:
        nxp, nyp, nzp = self.nelx + 1, self.nely + 1, self.nelz + 1
        return np.arange(nxp * nyp * nzp, dtype=np.int64).reshape(nzp, nyp, nxp)

    def x_min_face_nodes(self) -> np.ndarray:
        return self._node_grid()[:, :, 0].ravel()

    def x_max_face_nodes(self) -> np.ndarray:
        return self._node_grid()[:, :, -1].ravel()

    def y_min_face_nodes(self) -> np.ndarray:
        return self._node_grid()[:, 0, :].ravel()

    def y_max_face_nodes(self) -> np.ndarray:
        return self._node_grid()[:, -1, :].ravel()

    def z_min_face_nodes(self) -> np.ndarray:
        return self._node_grid()[0].ravel()

    def z_max_face_nodes(self) -> np.ndarray:
        return self._node_grid()[-1].ravel()


def structured_hex8(nelx: int, nely: int, nelz: int, thickness: float = 1.0) -> Hex8Mesh:
    """Regular grid of unit-cube hex8 elements per the binding conventions."""
    iz, iy, ix = np.meshgrid(
        np.arange(nelz + 1),
        np.arange(nely + 1),
        np.arange(nelx + 1),
        indexing="ij",
    )
    coords = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.float64)
    econn = _structured_econn(nelx, nely, nelz)
    return Hex8Mesh(
        nelx=nelx, nely=nely, nelz=nelz, coords=coords, econn=econn, thickness=thickness
    )


def distorted_hex8(
    nelx: int, nely: int, nelz: int, amplitude: float = 0.2, seed: int = 0, thickness: float = 1.0
) -> Hex8Mesh:
    """Structured mesh with deterministic jitter of INTERIOR nodes only.

    amplitude < ~0.3 keeps all element Jacobians positive on unit cubes.
    Boundary nodes stay put so BC face selectors remain valid.
    """
    assert 0.0 <= amplitude < 0.3, "amplitude must keep Jacobians positive"
    mesh = structured_hex8(nelx, nely, nelz, thickness)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-amplitude, amplitude, size=mesh.coords.shape)
    ix = mesh.coords[:, 0]
    iy = mesh.coords[:, 1]
    iz = mesh.coords[:, 2]
    interior = (ix > 0) & (ix < nelx) & (iy > 0) & (iy < nely) & (iz > 0) & (iz < nelz)
    coords = mesh.coords.copy()
    coords[interior] += jitter[interior]
    return Hex8Mesh(
        nelx=nelx, nely=nely, nelz=nelz, coords=coords, econn=mesh.econn.copy(), thickness=thickness
    )


def dof_ids_3d(node_ids: np.ndarray) -> np.ndarray:
    """All (ux, uy, uz) dof ids for the given node ids, interleaved."""
    node_ids = np.asarray(node_ids, dtype=np.int64).ravel()
    return np.stack([3 * node_ids, 3 * node_ids + 1, 3 * node_ids + 2], axis=1).ravel()
