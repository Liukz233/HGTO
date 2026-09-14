"""Geometric preconditioning for compact Cartesian Q4 domains with cutouts.

The embedding is only a multigrid workspace. Absent cells have exactly zero
stiffness, and absent nodes are masked. The physical graph operator, its loads,
and its certified residual remain on the original compact mesh.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class CartesianEmbedding:
    nx: int
    ny: int
    nodes: np.ndarray
    elements: np.ndarray

    @classmethod
    def from_mesh(cls, mesh):
        xy = np.asarray(mesh.coords)
        cells = np.asarray(mesh.econn)
        if xy.shape[1] != 2 or cells.shape[1] != 4:
            raise ValueError("Cartesian embedding requires Q4 cells")
        axes = [np.unique(xy[:, d]) for d in range(2)]
        for axis in axes:
            if len(axis) < 3 or not np.allclose(
                np.diff(axis), np.diff(axis)[0], rtol=1e-8, atol=1e-12
            ):
                raise ValueError("Mesh is not a subset of a uniform Cartesian grid")
        nx, ny = len(axes[0]) - 1, len(axes[1]) - 1
        if (nx + 1) * (ny + 1) > 4 * len(xy):
            raise ValueError("Cartesian embedding would have excessive empty workspace")
        ix = np.searchsorted(axes[0], xy[:, 0])
        iy = np.searchsorted(axes[1], xy[:, 1])
        nodes = iy * (nx + 1) + ix
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Duplicate Cartesian nodes")
        mapped = nodes[cells]
        corner = mapped[:, 0]
        expected = np.stack([corner, corner + 1, corner + nx + 2, corner + nx + 1], axis=1)
        if not np.array_equal(mapped, expected):
            raise ValueError("Cells must be Cartesian BL, BR, TR, TL quads")
        ex, ey = ix[cells[:, 0]], iy[cells[:, 0]]
        if np.any(ex >= nx) or np.any(ey >= ny):
            raise ValueError("Invalid Cartesian cell")
        elements = ey * nx + ex
        if len(np.unique(elements)) != len(elements):
            raise ValueError("Duplicate Cartesian cells")
        return cls(nx, ny, nodes, elements)


class MaskedMGHierarchy:
    def __init__(self, operator, youngs):
        from .mgcg import MGHierarchy, _structured_econn

        emb = operator.cartesian_embedding
        device, dtype = operator.device, operator.dtype
        node_ids = torch.as_tensor(emb.nodes, device=device)
        self.dof_ids = (2 * node_ids[:, None] + torch.arange(2, device=device)).reshape(-1)
        self.n_dof = 2 * (emb.nx + 1) * (emb.ny + 1)
        free = torch.zeros(self.n_dof, dtype=torch.bool, device=device)
        free[self.dof_ids] = operator.free_dof_mask
        matrices = torch.zeros((emb.nx * emb.ny, 8, 8), device=device, dtype=dtype)
        matrices[torch.as_tensor(emb.elements, device=device)] = (
            operator.Ke0 * youngs[:, None, None]
        )
        self.hierarchy = MGHierarchy(
            operator.Ke0,
            youngs,
            _structured_econn(emb.nx, emb.ny, device),
            free,
            emb.nx,
            emb.ny,
            element_matrices=matrices,
            coarsest_dtype=operator.mgcg_coarsest_dtype,
            semi_coarsen=operator.mgcg_semi_coarsen,
        )
        self.coarsest_fp32_fallback = self.hierarchy.coarsest_fp32_fallback

    def apply(self, rhs):
        full = rhs.new_zeros(self.n_dof)
        full[self.dof_ids] = rhs
        return self.hierarchy.apply(full)[self.dof_ids]

    def stamp(self):
        return self.hierarchy.stamp() | {
            "embedding": "Cartesian cutout; zero absent-cell stiffness",
            "physical_dofs": int(self.dof_ids.numel()),
            "workspace_dofs": self.n_dof,
        }
