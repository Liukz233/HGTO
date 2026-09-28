"""Linear three-node triangles with counter-clockwise connectivity.

The mesh reuses the Q4 container: node coordinates, connectivity and
interleaved displacement DOFs (2 * node, 2 * node + 1). Element areas follow
from the same shoelace formula. Triangles never form a structured grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .q4 import Q4Mesh


@dataclass
class Tri3Mesh(Q4Mesh):
    @property
    def is_regular(self) -> bool:
        return False


def triangle_unit_stiffness(mesh, nu=0.3):
    """Plane-stress constant-strain element matrices for E=1, shape (Ne, 6, 6)."""
    p = mesh.coords[mesh.econn]
    x, y = p[..., 0], p[..., 1]
    twice = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    if np.any(twice <= 0):
        raise ValueError("Tri3 cells require positive oriented area")
    dx = (
        np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1) / twice[:, None]
    )
    dy = (
        np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1) / twice[:, None]
    )
    B = np.zeros((len(p), 3, 6))
    B[:, 0, 0::2] = dx
    B[:, 1, 1::2] = dy
    B[:, 2, 0::2] = dy
    B[:, 2, 1::2] = dx
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)
    return np.einsum("eia,ij,ejb,e->eab", B, D, B, 0.5 * twice * mesh.thickness)
