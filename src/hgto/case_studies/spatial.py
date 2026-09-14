"""Spatial Hex8 benchmarks with finite traction and support patches.

Coordinates follow the package convention: x is length, y is height, and z
is depth.  Forces are consistent Q4-face integrals of uniform tractions.
There are no imposed symmetry constraints and no prescribed solid elements.
"""

import numpy as np

from hgto.fem.mesh.hex8 import structured_hex8
from hgto.linear3d.problems import Problem3DSpec


def _face_patch(mesh, axis, position, limits):
    """Return face cells fully inside an axis-aligned rectangular patch."""
    tangents = [a for a in range(3) if a != axis]
    xyz = mesh.coords
    local_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    face_cells = []
    for local in local_faces:
        faces = mesh.econn[:, local]
        points = xyz[faces]
        keep = np.all(np.isclose(points[:, :, axis], position), axis=1)
        for a, (lo, hi) in zip(tangents, limits):
            keep &= np.all(
                (points[:, :, a] >= lo - 1e-10) & (points[:, :, a] <= hi + 1e-10), axis=1
            )
        if keep.any():
            face_cells.extend(faces[keep].tolist())
    faces = np.asarray(face_cells, dtype=np.int64).reshape(-1, 4)
    if not len(faces):
        raise ValueError("A boundary patch must contain at least one face cell")
    return faces


def _patch(mesh, axis, position, limits, label):
    faces = _face_patch(mesh, axis, position, limits)
    return dict(
        label=label,
        axis=axis,
        position=float(position),
        limits=[list(map(float, p)) for p in limits],
        face_cells=faces.tolist(),
        nodes=np.unique(faces).tolist(),
        area=float(len(faces)),
    )


def _traction(force, patch, resultant):
    faces = np.asarray(patch["face_cells"])
    total = np.asarray(resultant, dtype=float)
    nodal = total / (4 * len(faces))
    for component in range(3):
        np.add.at(force, 3 * faces.ravel() + component, nodal[component])
    patch["resultant"] = total.tolist()


def _centered_limits(n, halfwidth):
    left = int(round(n / 2 - halfwidth))
    return (max(0, left), min(n, left + 2 * halfwidth))


def make_3d_case(family="four_foot_support", shape=None, volfrac=None, rmin=1.6):
    """Return an existing-compatible ``(Problem3DSpec, Hex8Mesh)`` pair.

    ``cantilever_3d``: A conventional 3:1:1 box with its left face clamped
    and a central finite downward traction patch on its right face.
    ``four_foot_support``: Four square base clamps and a central top pressure pad,
    total downward load 1.  ``torsion_member``: Left face clamped, two opposed
    end-face traction patches producing a unit moment about x, with zero
    resultant force.  All patch bounds align with finite-element faces.
    """
    defaults = {
        "cantilever_3d": ((48, 16, 16), 0.30),
        "four_foot_support": ((24, 32, 24), 0.18),
        "torsion_member": ((32, 24, 24), 0.20),
    }
    if family not in defaults:
        raise ValueError(f"Unknown spatial benchmark: {family}")
    default_shape, default_volume = defaults[family]
    shape = tuple(int(n) for n in (shape or default_shape))
    if len(shape) != 3 or min(shape) < 8:
        raise ValueError("Spatial case dimensions must each be at least 8 elements")
    nx, ny, nz = shape
    mesh = structured_hex8(nx, ny, nz)
    force = np.zeros(mesh.n_dof)
    supports, loads = [], []
    if family == "cantilever_3d":
        supports.append(_patch(mesh, 0, 0, ((0, ny), (0, nz)), "Wall mount"))
        halfwidth = max(1, int(round(min(ny, nz) / 16)))
        pad = _patch(
            mesh,
            0,
            nx,
            (_centered_limits(ny, halfwidth), _centered_limits(nz, halfwidth)),
            "Tip pad",
        )
        _traction(force, pad, [0, -1, 0])
        loads.append(pad)
        note = (
            "Left face x=0 clamped in all three directions; a finite "
            "central patch on the right face x=L carries uniform downward "
            "traction with resultant (0,-1,0)."
        )
    elif family == "four_foot_support":
        foot_width = max(2, int(round(min(nx, nz) / 8)))
        for i, xr in enumerate(((0, foot_width), (nx - foot_width, nx))):
            for j, zr in enumerate(((0, foot_width), (nz - foot_width, nz))):
                supports.append(_patch(mesh, 1, 0, (xr, zr), f"Foot {2 * i + j + 1}"))
        halfwidth = max(1, int(round(min(nx, nz) / 12)))
        pad = _patch(
            mesh,
            1,
            ny,
            (_centered_limits(nx, halfwidth), _centered_limits(nz, halfwidth)),
            "Top pad",
        )
        _traction(force, pad, [0, -1, 0])
        loads.append(pad)
        note = (
            "Four separated square patches on y=0 clamped in all three "
            "directions; central top pad at y=H carries uniform downward "
            "traction of resultant (0,-1,0)."
        )
    else:
        supports.append(_patch(mesh, 0, 0, ((0, ny), (0, nz)), "Wall mount"))
        width = max(2, int(round(nz / 8)))
        halfheight = max(1, int(round(ny / 12)))
        yrange = _centered_limits(ny, halfheight)
        # Each resultant is 1/separation: the resulting moment M_x is 1.
        magnitude = 1.0 / (nz - width)
        low = _patch(mesh, 0, nx, (yrange, (0, width)), "Positive-y pad")
        high = _patch(mesh, 0, nx, (yrange, (nz - width, nz)), "Negative-y pad")
        _traction(force, low, [0, magnitude, 0])
        _traction(force, high, [0, -magnitude, 0])
        loads.extend((low, high))
        note = (
            "Left face x=0 clamped in all three directions; two finite "
            "patches at x=L carry opposed uniform y tractions. The "
            "resultant force is zero and moment is (1,0,0)."
        )
    nodes = np.unique(np.concatenate([np.asarray(p["nodes"]) for p in supports]))
    fixed = (3 * nodes[:, None] + np.arange(3)).ravel()
    fn = force.reshape(-1, 3)
    volume = default_volume if volfrac is None else float(volfrac)
    if not 0 < volume < 1 or rmin <= 0:
        raise ValueError("Require 0 < volume fraction < 1 and positive filter radius")
    extra = dict(
        family=family,
        shape=list(shape),
        dimension_order=["x", "y", "z"],
        support_patches=supports,
        load_patches=loads,
        load_nodes=np.flatnonzero(np.linalg.norm(fn, axis=1) > 0).tolist(),
        force_resultant=fn.sum(0).tolist(),
        moment_about_origin=np.cross(mesh.coords, fn).sum(0).tolist(),
        load_discretization="Consistent Q4-face integral of uniform traction",
        target_volume=volume,
        filter_radius=float(rmin),
        prescribed_solid_elements=False,
        symmetry_constraints=False,
    )
    problem = Problem3DSpec(
        f"{family}_{nx}x{ny}x{nz}",
        nx,
        ny,
        nz,
        volume,
        float(rmin),
        np.sort(fixed),
        force,
        note,
        extra,
    )
    return problem, mesh
