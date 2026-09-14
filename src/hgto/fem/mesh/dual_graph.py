"""Element adjacency for the density graph.

Q4 elements connect across full edges; Hex8 elements connect across full
faces. Both edge directions are returned. Edge features store the centroid
difference, centroid distance, and shared edge length or face area."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hgto.fem.mesh.hex8 import Hex8Mesh
from hgto.fem.mesh.q4 import Q4Mesh

# Local element edges CCW (pairs of local-node indices).
_LOCAL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))

# Local hex8 faces (cyclic quads of local-node indices, design_graph_3d).
_LOCAL_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def build_element_dual_graph(mesh: Q4Mesh | Hex8Mesh) -> Dict[str, np.ndarray]:
    """Returns dict(edge_index (2, E) int64, edge_attr (E, 4|5) float64)."""
    if mesh.econn.shape[1] == 8:
        return _hex8_dual_graph(mesh)
    face_owner: Dict[Tuple[int, int], int] = {}
    pairs = []  # (elem_i, elem_j, shared_len)
    coords = mesh.coords
    for e in range(mesh.n_elements):
        nodes = mesh.econn[e]
        for a, b in _LOCAL_EDGES:
            key = (int(min(nodes[a], nodes[b])), int(max(nodes[a], nodes[b])))
            if key in face_owner:
                other = face_owner.pop(key)
                length = float(np.linalg.norm(coords[key[0]] - coords[key[1]]))
                pairs.append((other, e, length))
            else:
                face_owner[key] = e

    if not pairs:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, 4), dtype=np.float64),
        }

    cent = mesh.element_centroids()
    src, dst, attr = [], [], []
    for i, j, shared_len in pairs:
        for s, d in ((i, j), (j, i)):
            delta = cent[d] - cent[s]
            src.append(s)
            dst.append(d)
            attr.append([delta[0], delta[1], float(np.linalg.norm(delta)), shared_len])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attr, dtype=np.float64)
    order = np.lexsort((edge_index[1], edge_index[0]))
    return {"edge_index": edge_index[:, order], "edge_attr": edge_attr[order]}


def _quad_face_area(p: np.ndarray) -> float:
    """Area of a possibly non-planar quad (4, 3) given in CYCLIC corner order.

    Split into two triangles by the 0-2 diagonal of the cycle and sum.
    """
    t1 = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    t2 = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
    return float(t1 + t2)


def _hex8_dual_graph(mesh: Hex8Mesh) -> Dict[str, np.ndarray]:
    """Hex8 path: shared-face adjacency, 5-column edge_attr.

    The shared-face area uses the first-seen (owner) element's cyclic face
    order for the triangle split — deterministic given the element ordering.
    """
    face_owner: Dict[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...]]] = {}
    pairs = []  # (elem_i, elem_j, shared face area)
    coords = mesh.coords
    for e in range(mesh.n_elements):
        nodes = mesh.econn[e]
        for face in _LOCAL_FACES:
            cycle = tuple(int(nodes[a]) for a in face)
            key = tuple(sorted(cycle))
            if key in face_owner:
                other, owner_cycle = face_owner.pop(key)
                area = _quad_face_area(coords[list(owner_cycle)])
                pairs.append((other, e, area))
            else:
                face_owner[key] = (e, cycle)

    if not pairs:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, 5), dtype=np.float64),
        }

    cent = mesh.element_centroids()
    src, dst, attr = [], [], []
    for i, j, area in pairs:
        for s, d in ((i, j), (j, i)):
            delta = cent[d] - cent[s]
            src.append(s)
            dst.append(d)
            attr.append([delta[0], delta[1], delta[2], float(np.linalg.norm(delta)), area])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attr, dtype=np.float64)
    order = np.lexsort((edge_index[1], edge_index[0]))
    return {"edge_index": edge_index[:, order], "edge_attr": edge_attr[order]}
