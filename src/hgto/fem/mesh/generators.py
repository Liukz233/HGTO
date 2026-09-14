"""Graded, mapped and cut-out quadrilateral mesh generators."""

from __future__ import annotations

import operator
from typing import Callable

import numpy as np

from hgto.fem.mesh.q4 import Q4Mesh, structured_q4
from hgto.fem.mesh.quadrature import q4_reference_tables

CoordinateMap = Callable[[np.ndarray], np.ndarray]


def _positive_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("{} must be an integer".format(name)) from error
    if result <= 0:
        raise ValueError("{} must be positive".format(name))
    return int(result)


def _graded_axis(n_elements: int, ratio: float, name: str) -> np.ndarray:
    ratio_value = float(ratio)
    if not np.isfinite(ratio_value) or ratio_value < 1.0:
        raise ValueError("{} must be finite and at least 1.0".format(name))
    if ratio_value == 1.0:
        return np.arange(n_elements + 1, dtype=np.float64)

    # ``ratio`` is the common ratio between consecutive cell widths.  The
    # shifted exponents avoid overflow; normalization retains domain length.
    exponents = np.arange(n_elements, dtype=np.float64) - float(n_elements - 1)
    widths = np.exp(np.log(ratio_value) * exponents)
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        raise ValueError("{} produces unresolved cell widths".format(name))
    widths *= float(n_elements) / float(np.sum(widths))
    coordinates = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(widths, dtype=np.float64)]
    )
    coordinates[-1] = float(n_elements)
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("{} produces unresolved cell widths".format(name))
    return coordinates


def _assert_positive_jacobians(mesh: Q4Mesh) -> None:
    gradients = q4_reference_tables()["dN_dxi"]
    element_coords = mesh.coords[mesh.econn]
    jacobians = np.einsum("eai,gaj->egij", element_coords, gradients)
    determinants = np.linalg.det(jacobians)
    assert np.all(np.isfinite(determinants)) and np.all(determinants > 0.0), (
        "coordinate mapping produced a non-positive Gauss-point Jacobian"
    )


def graded_q4(nelx: int, nely: int, ratio_x: float = 1.0, ratio_y: float = 1.0) -> Q4Mesh:
    """Return a complete Q4 grid with geometrically growing cell widths.

    ``ratio_x`` and ``ratio_y`` are adjacent-cell growth ratios.  The first
    cells at ``x=0`` and ``y=0`` are finest, and the physical bounding box
    remains ``[0, nelx] x [0, nely]``.
    """
    nx = _positive_int(nelx, "nelx")
    ny = _positive_int(nely, "nely")
    x = _graded_axis(nx, ratio_x, "ratio_x")
    y = _graded_axis(ny, ratio_y, "ratio_y")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    base = structured_q4(nx, ny)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    mesh = Q4Mesh(nx, ny, coords, base.econn.copy())
    _assert_positive_jacobians(mesh)
    return mesh


def mapped_q4(nelx: int, nely: int, mapping: CoordinateMap) -> Q4Mesh:
    """Apply ``mapping`` to a complete unit-square-parametric Q4 grid.

    The callable receives an ``(Nn, 2)`` float64 array whose columns range
    from zero to one and must return an array with the same shape.
    """
    nx = _positive_int(nelx, "nelx")
    ny = _positive_int(nely, "nely")
    if not callable(mapping):
        raise TypeError("mapping must be callable")
    base = structured_q4(nx, ny)
    parametric = base.coords / np.array([float(nx), float(ny)], dtype=np.float64)
    coords = np.asarray(mapping(parametric.copy()), dtype=np.float64)
    if coords.shape != parametric.shape:
        raise ValueError("mapping must return coordinates with shape (Nn, 2)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("mapping returned non-finite coordinates")
    mesh = Q4Mesh(nx, ny, coords.copy(), base.econn.copy())
    _assert_positive_jacobians(mesh)
    return mesh


def annulus_sector(r_in: float, r_out: float, angle: float) -> CoordinateMap:
    """Return a positive-orientation map for an annular sector.

    Parametric ``x`` runs radially from ``r_in`` to ``r_out`` and parametric
    ``y`` runs counter-clockwise from angle zero to ``angle``.
    """
    inner = float(r_in)
    outer = float(r_out)
    sector_angle = float(angle)
    if not (np.isfinite(inner) and np.isfinite(outer) and np.isfinite(sector_angle)):
        raise ValueError("annulus parameters must be finite")
    if inner <= 0.0 or outer <= inner:
        raise ValueError("annulus radii require 0 < r_in < r_out")
    if not 0.0 < sector_angle <= 2.0 * np.pi:
        raise ValueError("angle must lie in (0, 2*pi]")

    def mapping(parametric: np.ndarray) -> np.ndarray:
        points = np.asarray(parametric, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("annulus map expects coordinates with shape (Nn, 2)")
        radius = inner + (outer - inner) * points[:, 0]
        theta = sector_angle * points[:, 1]
        return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1).astype(np.float64)

    return mapping


def lbracket_q4(n: int, notch_frac: float = 0.4) -> Q4Mesh:
    """Return an ``n``-square grid with a top-right square notch removed.

    ``notch_frac`` is the removed square's side as a fraction of the bounding
    side.  Because the boundary follows element edges, its cell count is
    rounded to nearest via ``floor(n * notch_frac + 0.5)``.  Kept elements
    retain row-major order; used nodes are compacted in ascending old-node
    order, making numbering deterministic and orphan-free.
    """
    size = _positive_int(n, "n")
    fraction = float(notch_frac)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("notch_frac must lie in (0, 1)")
    notch_cells = int(np.floor(float(size) * fraction + 0.5))
    if notch_cells <= 0 or notch_cells >= size:
        raise ValueError("notch_frac must resolve to between 1 and n-1 cells")

    base = structured_q4(size, size)
    ex, ey = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    cut_start = size - notch_cells
    keep = ~((ex.ravel() >= cut_start) & (ey.ravel() >= cut_start))
    kept_connectivity = base.econn[keep]
    used_nodes = np.unique(kept_connectivity.reshape(-1))
    old_to_new = np.full(base.n_nodes, -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    compact_connectivity = old_to_new[kept_connectivity]
    mesh = Q4Mesh(
        nelx=size,
        nely=size,
        coords=base.coords[used_nodes].copy(),
        econn=compact_connectivity.astype(np.int64, copy=False),
    )
    _assert_positive_jacobians(mesh)
    return mesh


def staircase_slit_q4(nelx: int, nely: int, runs: list[tuple[int, int, int]]) -> Q4Mesh:
    """Structured grid with a one-element-wide slit carved along a staircase.

    ``runs`` is a list of ``(row, x_start, x_end)`` horizontal cuts (element
    coordinates, ``x_end`` exclusive), ordered left to right and contiguous:
    each run must start where the previous one ended (``x_start[i+1] ==
    x_end[i]``). At every transition the connecting jog COLUMN is carved too
    (rows between the two runs, inclusive), so the slit is a sealed polyline
    — the two banks connect only around the slit's open ends. A single run
    degenerates to a straight slit (the E15a coordinate-easy control).

    One element of slit width is a deliberate contract: facing elements
    across the slit sit two cell-heights apart in centroid distance, which
    keeps the geometric density filter (rmin 1.5 house default) from
    coupling across the cut. Kept elements retain row-major order; nodes are
    compacted ascending (the lbracket_q4 conventions).
    """
    nx = _positive_int(nelx, "nelx")
    ny = _positive_int(nely, "nely")
    if not runs:
        raise ValueError("runs must contain at least one (row, x_start, x_end)")
    removed: set[tuple[int, int]] = set()
    previous_row = None
    previous_end = None
    for index, (row, x_start, x_end) in enumerate(runs):
        row, x_start, x_end = int(row), int(x_start), int(x_end)
        if not 0 < row < ny - 1:
            raise ValueError("slit rows must be interior (0 < row < nely-1)")
        if not 0 <= x_start < x_end <= nx:
            raise ValueError("run x-range must satisfy 0 <= x_start < x_end <= nelx")
        if index > 0:
            if x_start != previous_end:
                raise ValueError("runs must be contiguous (x_start == previous x_end)")
            if row == previous_row:
                raise ValueError("consecutive runs must change row")
            low, high = sorted((previous_row, row))
            for jog_row in range(low, high + 1):
                removed.add((x_start, jog_row))
        for ex in range(x_start, x_end):
            removed.add((ex, row))
        previous_row, previous_end = row, x_end
    if len(removed) >= nx * ny:
        raise ValueError("slit removes the entire grid")

    base = structured_q4(nx, ny)
    keep = np.ones(nx * ny, dtype=bool)
    for ex, ey in removed:
        keep[ey * nx + ex] = False
    kept_connectivity = base.econn[keep]
    used_nodes = np.unique(kept_connectivity.reshape(-1))
    old_to_new = np.full(base.n_nodes, -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    mesh = Q4Mesh(
        nelx=nx,
        nely=ny,
        coords=base.coords[used_nodes].copy(),
        econn=old_to_new[kept_connectivity].astype(np.int64, copy=False),
    )
    _assert_positive_jacobians(mesh)
    return mesh
