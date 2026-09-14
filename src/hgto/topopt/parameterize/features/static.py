"""Coordinate normalization for graph density inputs."""

import numpy as np
import torch
from hgto.fem.mesh.q4 import Q4Mesh
from hgto.fem.mesh.hex8 import Hex8Mesh


def unit_bbox_centroids(mesh: Q4Mesh | Hex8Mesh) -> torch.Tensor:
    """Element centroids normalized to ``[0, 1]^d`` by the domain bbox.

    Returns a detached
    CPU float64 tensor of shape ``(Ne, n_dim)``.
    """
    centroids = mesh.element_centroids()
    lower = np.min(mesh.coords, axis=0)
    extent = np.max(mesh.coords, axis=0) - lower
    if np.any(extent <= 0.0):
        raise ValueError("mesh bounding box must have positive extent")
    normalized = (centroids - lower) / extent
    return torch.from_numpy(np.ascontiguousarray(normalized)).detach()
