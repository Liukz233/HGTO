"""Canonical L-bracket and a cantilever plate with a circular clearance hole."""

import json
from pathlib import Path

import numpy as np

from hgto.fem.mesh.q4 import Q4Mesh
from hgto.domains.geometry import quads_from_triangles


def _boundary_edges(cells):
    edges = np.sort(np.concatenate([cells[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _loaded_edge_forces(points, cells, predicate, resultant):
    edges = _boundary_edges(cells)
    edges = edges[np.all(predicate(points[edges]), axis=1)]
    if not len(edges):
        raise ValueError("Load port contains no full boundary edges")
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    force = np.zeros(2 * len(points))
    resultant = np.asarray(resultant)
    for component in range(2):
        weights = np.repeat(length * resultant[component] / (2 * length.sum()), 2)
        np.add.at(force, 2 * edges.ravel() + component, weights)
    return force, edges


def _l_bracket(resolution):
    n = int(resolution)
    if n < 20 or n % 20:
        raise ValueError("L-bracket resolution must be a multiple of 20, at least 20")
    x, y = np.meshgrid(np.linspace(0, 2, n + 1), np.linspace(0, 2, n + 1))
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    ex, ey = np.meshgrid(np.arange(n), np.arange(n))
    keep = (ex < n // 2) | (ey < n // 2)
    a = (ey * (n + 1) + ex)[keep]
    cells = np.stack([a, a + 1, a + n + 2, a + n + 1], axis=1)
    used = np.unique(cells)
    index = np.full(len(points), -1)
    index[used] = np.arange(len(used))
    points, cells = points[used], index[cells]
    support_nodes = np.flatnonzero(np.isclose(points[:, 1], 2))
    predicate = lambda p: np.isclose(p[..., 0], 2) & (p[..., 1] >= 0.8 - 1e-9)
    force, loaded_edges = _loaded_edge_forces(points, cells, predicate, [0, -1])
    meta = dict(
        name="l_bracket_domain",
        family="irregular_domain",
        bounding_box=[2.0, 2.0],
        cutout=[1.0, 1.0, 2.0, 2.0],
        nominal_mesh_size=2 / n,
        target_volume=0.40,
        filter_radius=0.09,
        mesh_algorithm="Conforming compact Cartesian Q4 mesh",
        supports="Top mounting edge y=2, 0<=x<=1: ux=uy=0",
        loads="Uniform downward traction on x=2, 0.8<=y<=1, resultant Fy=-1",
        support_bounds=[[0.0, 2.0], [1.0, 2.0]],
        load_bounds=[[2.0, 0.8], [2.0, 1.0]],
        excluded_geometry="Upper-right unit square is absent from the mesh",
    )
    return points, cells, support_nodes, force, loaded_edges, meta


def _perforated_bracket(mesh_size):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RandomSeed", 17)
        gmsh.model.add("perforated_bracket")
        geo = gmsh.model.geo
        # Right-edge break points enforce the exact finite load-port bounds.
        boundary = [(0, 0), (3, 0), (3, 0.9), (3, 1.1), (3, 2), (0, 2)]
        tags = [geo.addPoint(x, y, 0, mesh_size) for x, y in boundary]
        outer = geo.addCurveLoop(
            [geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        )
        center = (1.5, 1.0)
        radius = 0.5
        origin = geo.addPoint(*center, 0, mesh_size)
        circle = [
            geo.addPoint(
                center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), 0, mesh_size
            )
            for a in np.arange(4) * np.pi / 2
        ]
        hole = geo.addCurveLoop(
            [geo.addCircleArc(circle[i], origin, circle[(i + 1) % 4]) for i in range(4)]
        )
        geo.addPlaneSurface([outer, hole])
        geo.synchronize()
        gmsh.model.mesh.generate(2)
        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(xyz).reshape(-1, 3)[:, :2]
        index = {int(t): i for i, t in enumerate(node_tags)}
        types, _, connectivity = gmsh.model.mesh.getElements(2)
        if list(types) != [2]:
            raise RuntimeError("Expected a triangular Gmsh surface mesh")
        triangles = np.asarray([index[int(n)] for n in connectivity[0]]).reshape(-1, 3)
        used = np.unique(triangles)
        remap = np.full(len(points), -1)
        remap[used] = np.arange(len(used))
        points, cells = quads_from_triangles(points[used], remap[triangles])
    finally:
        gmsh.finalize()
    support_nodes = np.flatnonzero(np.isclose(points[:, 0], 0))
    predicate = lambda p: (
        np.isclose(p[..., 0], 3) & (p[..., 1] >= 0.9 - 1e-9) & (p[..., 1] <= 1.1 + 1e-9)
    )
    force, loaded_edges = _loaded_edge_forces(points, cells, predicate, [0, -1])
    meta = dict(
        name="perforated_bracket",
        family="unstructured_mesh",
        bounding_box=[3.0, 2.0],
        hole_center=[1.5, 1.0],
        hole_radius=0.5,
        nominal_mesh_size=float(mesh_size),
        target_volume=0.40,
        filter_radius=0.12,
        mesh_algorithm="Gmsh frontal-Delaunay triangles, conforming three-Q4 subdivision",
        supports="Left mounting edge x=0, 0<=y<=2: ux=uy=0",
        loads="Uniform downward traction on x=3, 0.9<=y<=1.1, resultant Fy=-1",
        support_bounds=[[0.0, 0.0], [0.0, 2.0]],
        load_bounds=[[3.0, 0.9], [3.0, 1.1]],
        excluded_geometry="Circular central clearance hole, traction-free boundary",
    )
    return points, cells, support_nodes, force, loaded_edges, meta


def make_domain_case(family="l_bracket_domain", resolution=80, mesh_size=0.10):
    """Return ``coords, econn, fixed_dofs, force, metadata``."""
    if family == "l_bracket_domain":
        points, cells, support_nodes, force, loaded_edges, meta = _l_bracket(resolution)
    elif family == "perforated_bracket":
        points, cells, support_nodes, force, loaded_edges, meta = _perforated_bracket(mesh_size)
    else:
        raise ValueError(f"Unknown irregular benchmark: {family}")
    fixed = (2 * support_nodes[:, None] + np.arange(2)).ravel()
    mesh = Q4Mesh(0, 0, points, cells)
    area = mesh.element_volumes()
    meta.update(
        n_nodes=len(points),
        n_elements=len(cells),
        area=float(area.sum()),
        min_element_area=float(area.min()),
        max_element_area=float(area.max()),
        load_nodes=np.unique(loaded_edges).tolist(),
        loaded_edges=loaded_edges.tolist(),
        support_nodes=support_nodes.tolist(),
        force_resultant=force.reshape(-1, 2).sum(0).tolist(),
        load_discretization="Consistent line integral of uniform edge traction",
        prescribed_solid_elements=False,
        all_other_boundaries="Traction-free",
    )
    return points, cells, np.sort(fixed), force, meta


def save_domain_case(folder, family, **kwargs):
    """Write the portable format accepted by ``hgto.domains.load_case``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    coords, econn, fixed, force, meta = make_domain_case(family, **kwargs)
    mesh = Q4Mesh(0, 0, coords, econn)
    np.savez_compressed(
        folder / "geometry.npz",
        coords=coords,
        econn=econn,
        fixed_dofs=fixed,
        force=force,
        element_volumes=mesh.element_volumes(),
    )
    (folder / "case.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
