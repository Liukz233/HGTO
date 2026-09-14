"""Explicit, portable Q4 design problems; x-fastest cells, y upwards."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    name: str
    coords: np.ndarray
    cells: np.ndarray
    fixed: np.ndarray
    forces: np.ndarray
    volume_fraction: float
    filter_radius: float
    shape: tuple
    active_cells: np.ndarray
    description: str

    @property
    def centroids(self):
        return self.coords[self.cells].mean(1)

    @property
    def n_elements(self):
        return len(self.cells)

    @property
    def n_dofs(self):
        return 2 * len(self.coords)


def make_problem(family="cantilever", nx=120, ny=40, radius=None, volume=0.5):
    y, x = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    coords = np.c_[x.ravel(), y.ravel()].astype(float)
    ey, ex = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    n = ey.ravel() * (nx + 1) + ex.ravel()
    cells = np.c_[n, n + 1, n + nx + 2, n + nx + 1]
    active = np.arange(nx * ny)
    if family == "l_bracket":
        mask = ~((ex.ravel() >= nx // 2) & (ey.ravel() >= ny // 2))
        active = active[mask]
        cells = cells[mask]
    used, inv = np.unique(cells, return_inverse=True)
    old_to_new = np.full(len(coords), -1, dtype=int)
    old_to_new[used] = np.arange(len(used))
    cells = inv.reshape(-1, 4)
    coords = coords[used]

    def node(ix, iy):
        j = int(old_to_new[iy * (nx + 1) + ix])
        assert j >= 0
        return j

    def clamp(nodes):
        return np.sort(np.r_[2 * np.asarray(nodes), 2 * np.asarray(nodes) + 1])

    force = np.zeros(
        (
            2 * len(coords),
            3 if family == "bridge_multiload" else (2 if family == "multiload" else 1),
        )
    )
    if family in ["cantilever", "inclined", "multiload"]:
        fixed = clamp([node(0, j) for j in range(ny + 1)])
        if family == "cantilever":
            force[2 * node(nx, ny // 2) + 1, 0] = -1.0
            desc = "left edge clamped; downward unit force at right midpoint"
        elif family == "inclined":
            angle = np.deg2rad(25.0)
            force[2 * node(nx, 3 * ny // 4) : 2 * node(nx, 3 * ny // 4) + 2, 0] = [
                -np.sin(angle),
                -np.cos(angle),
            ]
            desc = "left edge clamped; unit load at right three-quarter height, 25 degrees inward from downward"
        else:
            force[2 * node(nx, 3 * ny // 4) + 1, 0] = -1.0
            force[2 * node(nx, ny // 4), 1] = -1.0
            force /= np.sqrt(2.0)
            desc = (
                "left edge clamped; equal-weight separate downward-upper and inward-lower end loads"
            )
    elif family == "mbb":
        fixed = np.sort(np.r_[[2 * node(0, j) for j in range(ny + 1)], 2 * node(nx, 0) + 1])
        force[2 * node(0, ny) + 1, 0] = -1.0
        desc = (
            "half-MBB: left ux symmetry, lower-right vertical roller, upper-left downward unit load"
        )
    elif family == "bridge_multiload":
        bearing_width = max(1, nx // 64)
        bearings = [node(i, 0) for i in range(bearing_width + 1)] + [
            node(i, 0) for i in range(nx - bearing_width, nx + 1)
        ]
        fixed = np.sort(np.r_[2 * node(0, 0), 2 * np.asarray(bearings) + 1])
        # Independent traffic positions, each a short distributed deck load.
        # sqrt(weight) encoding yields sum_l w_l f_l^T K^-1 f_l.
        halfwidth = max(1, nx // 64)
        for k, fraction in enumerate((0.25, 0.5, 0.75)):
            center = int(round(fraction * nx))
            nodes = [node(i, 0) for i in range(center - halfwidth, center + halfwidth + 1)]
            weights = np.ones(len(nodes))
            weights[[0, -1]] = 0.5
            force[2 * np.asarray(nodes) + 1, k] = -weights / (weights.sum() * np.sqrt(3.0))
        desc = "bridge: finite lower-end vertical bearing patches of width span/64, one left horizontal anchor; three independent equal-weight downward deck-patch loads at quarter, mid and three-quarter span; each unweighted resultant is -1"
    elif family == "bridge":
        fixed = np.array([2 * node(0, 0), 2 * node(0, 0) + 1, 2 * node(nx, 0) + 1])
        force[2 * node(nx // 2, ny) + 1, 0] = -1.0
        desc = "pin and roller at lower corners; downward unit load at upper midpoint"
    elif family == "l_bracket":
        fixed = clamp([node(i, ny) for i in range(nx // 2 + 1)])
        force[2 * node(nx, ny // 2) + 1, 0] = -1.0
        desc = "L-domain with upper-right quadrant removed; top edge clamped and outer horizontal tip loaded downward"
    else:
        raise ValueError(family)
    return Problem(
        f"{family}_{nx}x{ny}",
        coords,
        cells,
        fixed,
        force,
        volume,
        radius if radius is not None else 0.075 * ny,
        (ny, nx),
        active,
        desc,
    )
