"""Q4 meshes with counter-clockwise local nodes and interleaved displacement DOFs.

Structured grids use node_id = iy * (nelx + 1) + ix and
element_id = ey * nelx + ex. Local nodes run counter-clockwise from the
bottom-left corner; displacement DOFs are (2 * node_id, 2 * node_id + 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Q4Mesh:
    nelx: int
    nely: int
    coords: np.ndarray  # (Nn, 2) float64 node coordinates
    econn: np.ndarray  # (Ne, 4) int64, local nodes CCW from bottom-left
    thickness: float = 1.0

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self.econn.shape[0]

    @property
    def n_dof(self) -> int:
        return 2 * self.n_nodes

    @property
    def is_regular(self) -> bool:
        """Whether connectivity is the complete binding structured grid.

        Coordinates may be graded or smoothly mapped: this is deliberately a
        grid-completeness test, not a uniform-spacing test. It lets
        regular-only consumers reject compact cut-out/imported meshes without
        weakening support in mesh-agnostic mechanics code.
        """
        if self.nelx <= 0 or self.nely <= 0:
            return False
        if self.n_nodes != (self.nelx + 1) * (self.nely + 1):
            return False
        if self.n_elements != self.nelx * self.nely:
            return False
        ex, ey = np.meshgrid(np.arange(self.nelx), np.arange(self.nely), indexing="xy")
        n0 = ey.ravel() * (self.nelx + 1) + ex.ravel()
        expected = np.stack([n0, n0 + 1, n0 + self.nelx + 2, n0 + self.nelx + 1], axis=1).astype(
            np.int64
        )
        return bool(np.array_equal(self.econn, expected))

    def node_id(self, ix: int, iy: int) -> int:
        assert 0 <= ix <= self.nelx and 0 <= iy <= self.nely
        return iy * (self.nelx + 1) + ix

    def element_id(self, ex: int, ey: int) -> int:
        assert 0 <= ex < self.nelx and 0 <= ey < self.nely
        return ey * self.nelx + ex

    def element_centroids(self) -> np.ndarray:
        return self.coords[self.econn].mean(axis=1)

    def element_areas(self) -> np.ndarray:
        """Polygon (shoelace) areas — geometric, independent of any FEM code."""
        p = self.coords[self.econn]  # (Ne, 4, 2)
        x, y = p[..., 0], p[..., 1]
        xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
        return 0.5 * np.abs(np.sum(x * yn - xn * y, axis=1))

    def element_volumes(self) -> np.ndarray:
        return self.thickness * self.element_areas()

    # ---- boundary selectors (node ids) ----
    def left_edge_nodes(self) -> np.ndarray:
        return np.arange(self.nely + 1, dtype=np.int64) * (self.nelx + 1)

    def right_edge_nodes(self) -> np.ndarray:
        return self.left_edge_nodes() + self.nelx

    def bottom_edge_nodes(self) -> np.ndarray:
        return np.arange(self.nelx + 1, dtype=np.int64)

    def top_edge_nodes(self) -> np.ndarray:
        return self.bottom_edge_nodes() + self.nely * (self.nelx + 1)

    def bottom_row_elements(self) -> np.ndarray:
        return np.arange(self.nelx, dtype=np.int64)


def structured_q4(nelx: int, nely: int, thickness: float = 1.0) -> Q4Mesh:
    """Regular grid of unit-square Q4 elements per the binding conventions."""
    ix, iy = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1), indexing="xy")
    coords = np.stack([ix.ravel(), iy.ravel()], axis=1).astype(np.float64)

    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="xy")
    ex, ey = ex.ravel(), ey.ravel()
    n0 = ey * (nelx + 1) + ex  # bottom-left
    econn = np.stack([n0, n0 + 1, n0 + nelx + 2, n0 + nelx + 1], axis=1)
    return Q4Mesh(
        nelx=nelx, nely=nely, coords=coords, econn=econn.astype(np.int64), thickness=thickness
    )


def distorted_q4(
    nelx: int, nely: int, amplitude: float = 0.2, seed: int = 0, thickness: float = 1.0
) -> Q4Mesh:
    """Structured mesh with deterministic jitter of INTERIOR nodes only.

    amplitude < ~0.3 keeps all element Jacobians positive on unit squares.
    Boundary nodes stay put so BC selectors remain valid.
    """
    assert 0.0 <= amplitude < 0.3, "amplitude must keep Jacobians positive"
    mesh = structured_q4(nelx, nely, thickness)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-amplitude, amplitude, size=mesh.coords.shape)
    ix = mesh.coords[:, 0]
    iy = mesh.coords[:, 1]
    interior = (ix > 0) & (ix < nelx) & (iy > 0) & (iy < nely)
    coords = mesh.coords.copy()
    coords[interior] += jitter[interior]
    return Q4Mesh(nelx=nelx, nely=nely, coords=coords, econn=mesh.econn.copy(), thickness=thickness)


def dof_ids(node_ids: np.ndarray) -> np.ndarray:
    """All (ux, uy) dof ids for the given node ids, interleaved convention."""
    node_ids = np.asarray(node_ids, dtype=np.int64).ravel()
    return np.stack([2 * node_ids, 2 * node_ids + 1], axis=1).ravel()
