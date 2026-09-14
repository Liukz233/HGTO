"""Explicit structured Hex8 problem definitions."""

from dataclasses import dataclass, field
import numpy as np
from hgto.fem.mesh.hex8 import structured_hex8


@dataclass(frozen=True)
class Problem3DSpec:
    name: str
    nelx: int
    nely: int
    nelz: int
    volfrac: float
    rmin: float  # filter radius, element units
    fixed_dofs: np.ndarray  # (Nfix,) int64, canonical dof ids
    force: np.ndarray  # (Ndof,) float64
    bc_note: str = ""
    extra: dict = field(default_factory=dict)

    # ---- derived geometry -------------------------------------------------
    @property
    def n_elem(self) -> int:
        return self.nelx * self.nely * self.nelz

    @property
    def n_node(self) -> int:
        return (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)

    @property
    def n_dof(self) -> int:
        return 3 * self.n_node

    def node_id(self, ix: int, iy: int, iz: int) -> int:
        return iz * (self.nelx + 1) * (self.nely + 1) + iy * (self.nelx + 1) + ix

    def element_id(self, ex: int, ey: int, ez: int) -> int:
        return ez * self.nelx * self.nely + ey * self.nelx + ex

    def coords(self) -> np.ndarray:
        """(Nn, 3) nodal coordinates, canonical ordering (x fastest)."""
        iz, iy, ix = np.meshgrid(
            np.arange(self.nelz + 1),
            np.arange(self.nely + 1),
            np.arange(self.nelx + 1),
            indexing="ij",
        )
        return np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)], axis=1).astype(np.float64)

    def econn(self) -> np.ndarray:
        """(Ne, 8) connectivity: bottom face CCW from (ex,ey,ez), then top."""
        nxp = self.nelx + 1
        layer = nxp * (self.nely + 1)
        ez, ey, ex = np.meshgrid(
            np.arange(self.nelz), np.arange(self.nely), np.arange(self.nelx), indexing="ij"
        )
        n0 = ez.ravel() * layer + ey.ravel() * nxp + ex.ravel()
        bottom = np.stack([n0, n0 + 1, n0 + nxp + 1, n0 + nxp], axis=1)
        return np.concatenate([bottom, bottom + layer], axis=1).astype(np.int64)

    def element_centroids(self) -> np.ndarray:
        """(Ne, 3) element centers, canonical element ordering."""
        ez, ey, ex = np.meshgrid(
            np.arange(self.nelz), np.arange(self.nely), np.arange(self.nelx), indexing="ij"
        )
        return np.stack(
            [ex.reshape(-1) + 0.5, ey.reshape(-1) + 0.5, ez.reshape(-1) + 0.5], axis=1
        ).astype(np.float64)

    def rho_as_volume(self, rho: np.ndarray) -> np.ndarray:
        """(nelz, nely, nelx) voxel array; [ez, ey, ex] = element (ex,ey,ez).

        The canonical element id ez*nelx*nely + ey*nelx + ex is exactly the
        C-order ravel of this shape, so reshape is the whole mapping.
        """
        return np.asarray(rho, dtype=np.float64).reshape(self.nelz, self.nely, self.nelx)


def make_case(family="tip_cantilever", shape=None):
    if shape is None:
        shape = (40, 20, 16) if family != "bridge" else (48, 16, 16)
    nx, ny, nz = shape
    mesh = structured_hex8(nx, ny, nz)
    xyz = mesh.coords
    fixed_nodes = np.flatnonzero(xyz[:, 0] == 0)
    if family == "bridge":
        fixed_nodes = np.flatnonzero((xyz[:, 0] == 0) | (xyz[:, 0] == nx))
        # Two distinct finite load ports on the upper surface.
        loaded = (xyz[:, 1] == ny) & (abs(xyz[:, 0] - nx / 3) <= 1) & (abs(xyz[:, 2] - nz / 4) <= 1)
        loaded |= (
            (xyz[:, 1] == ny)
            & (abs(xyz[:, 0] - 2 * nx / 3) <= 1)
            & (abs(xyz[:, 2] - 3 * nz / 4) <= 1)
        )
        desc = "Both end faces fixed; two downward load patches at (L/3,H,B/4) and (2L/3,H,3B/4)"
    elif family == "tip_cantilever":
        loaded = (xyz[:, 0] == nx) & (xyz[:, 1] <= 1) & (abs(xyz[:, 2] - nz / 2) <= 1)
        desc = "Left face fixed; downward load on a small patch at the lower free-end edge"
    elif family == "edge_cantilever":
        loaded = (xyz[:, 0] == nx) & (xyz[:, 1] == ny // 2)
        desc = "Left face fixed; downward load distributed along the free-face horizontal midline"
    else:
        raise ValueError(family)
    load_nodes = np.flatnonzero(loaded)
    force = np.zeros(mesh.n_dof)
    force[3 * load_nodes + 1] = -1 / len(load_nodes)
    fixed = np.sort((3 * fixed_nodes[:, None] + np.arange(3)).ravel())
    volume = 0.3 if family != "bridge" else 0.25
    problem = Problem3DSpec(
        f"{family}_{nx}x{ny}x{nz}",
        nx,
        ny,
        nz,
        volume,
        2.0,
        fixed,
        force,
        desc,
        dict(
            family=family,
            load_nodes=load_nodes.tolist(),
            shape=[nx, ny, nz],
            total_load=1.0,
            n_load_nodes=len(load_nodes),
        ),
    )
    return problem, mesh
