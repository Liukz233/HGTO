"""Conforming irregular-domain and unstructured-quadrilateral mesh generators."""

import numpy as np


def quads_from_triangles(points, triangles):
    points = [list(p) for p in points]
    edge_mid = {}
    quads = []

    def mid(a, b):
        key = tuple(sorted((int(a), int(b))))
        if key not in edge_mid:
            edge_mid[key] = len(points)
            points.append(((np.array(points[a]) + points[b]) / 2).tolist())
        return edge_mid[key]

    for tri in triangles:
        a, b, c = map(int, tri)
        xy = np.array([points[a], points[b], points[c]])
        if np.linalg.det(np.stack((xy[1] - xy[0], xy[2] - xy[0]))) < 0:
            b, c = c, b
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        g = len(points)
        points.append(((np.array(points[a]) + points[b] + points[c]) / 3).tolist())
        quads += [[a, ab, g, ca], [b, bc, g, ab], [c, ca, g, bc]]
    return np.array(points), np.array(quads, dtype=np.int64)


def curved_arm(mesh_size=0.11):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.RandomSeed", 17)
    gmsh.model.add("curved_mounting_arm")
    geo = gmsh.model.geo
    o = geo.addPoint(0, 0, 0, mesh_size)
    a = geo.addPoint(2.6, 0, 0, mesh_size)
    b = geo.addPoint(0, 2.6, 0, mesh_size)
    c = geo.addPoint(0, 1.0, 0, mesh_size)
    d = geo.addPoint(1.0, 0, 0, mesh_size)
    edges = [
        geo.addCircleArc(a, o, b),
        geo.addLine(b, c),
        geo.addCircleArc(c, o, d),
        geo.addLine(d, a),
    ]
    loop = geo.addCurveLoop(edges)
    geo.addPlaneSurface([loop])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords).reshape(-1, 3)[:, :2]
    index = {int(t): i for i, t in enumerate(tags)}
    types, _, nodes = gmsh.model.mesh.getElements(2)
    assert set(types) == {2}, types
    triangles = np.array([index[int(n)] for n in nodes[0]]).reshape(-1, 3)
    used = np.unique(triangles)
    remap = np.full(len(coords), -1)
    remap[used] = np.arange(len(used))
    points, quads = quads_from_triangles(coords[used], remap[triangles])
    gmsh.finalize()
    fixed_nodes = np.flatnonzero(np.abs(points[:, 0]) < 1e-9)
    load_nodes = np.flatnonzero((np.abs(points[:, 1]) < 1e-9) & (points[:, 0] >= 2.3 - 1e-9))
    force = np.zeros(2 * len(points))
    force[2 * load_nodes + 1] = -1 / len(load_nodes)
    fixed = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    return (
        points,
        quads,
        fixed,
        force,
        dict(
            name="curved_mounting_arm",
            family="unstructured_mesh",
            inner_radius=1.0,
            outer_radius=2.6,
            angle_degrees=90,
            nominal_mesh_size=mesh_size,
            mesh_algorithm="Gmsh frontal Delaunay triangles, conforming three-Q4 subdivision",
            target_volume=0.45,
            filter_radius=0.14,
            load_nodes=load_nodes.tolist(),
            supports="Both displacements fixed on x=0 radial face",
            loads="Total downward unit force distributed on y=0, x in [2.3,2.6]",
        ),
    )


def stepped_bracket(nx=100, ny=80):
    # An offset service bracket, with a central upper-right clearance window.
    x, y = np.meshgrid(np.linspace(0, 2.5, nx + 1), np.linspace(0, 2.0, ny + 1))
    points = np.stack([x.ravel(), y.ravel()], 1)
    quads = []
    for j in range(ny):
        for i in range(nx):
            cx, cy = (i + 0.5) * 2.5 / nx, (j + 0.5) * 2.0 / ny
            if cx > 1.0 and cy > 0.75:
                continue
            a = j * (nx + 1) + i
            quads.append([a, a + 1, a + nx + 2, a + nx + 1])
    quads = np.array(quads)
    used = np.unique(quads)
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    points = points[used]
    quads = remap[quads]
    fixed_nodes = np.flatnonzero((np.abs(points[:, 1] - 2) < 1e-9) & (points[:, 0] <= 1 + 1e-9))
    load_nodes = np.flatnonzero(
        (np.abs(points[:, 0] - 2.5) < 1e-9) & (points[:, 1] >= 0.30) & (points[:, 1] <= 0.45)
    )
    fixed = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    force = np.zeros(2 * len(points))
    force[2 * load_nodes + 1] = -1 / len(load_nodes)
    return (
        points,
        quads,
        fixed,
        force,
        dict(
            name="offset_service_bracket",
            family="irregular_domain",
            bounding_box=[2.5, 2.0],
            clearance=[1.0, 0.75, 2.5, 2.0],
            nominal_mesh_size=2.5 / nx,
            mesh_algorithm="compact conforming Q4 grid on an L-shaped domain",
            target_volume=0.40,
            filter_radius=0.09,
            load_nodes=load_nodes.tolist(),
            supports="Both displacements fixed on upper mounting edge y=2, x in [0,1]",
            loads="Total downward unit force distributed on right port x=2.5, y in [0.3,0.45]",
        ),
    )
