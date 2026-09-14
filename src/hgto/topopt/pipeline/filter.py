"""Sparse, volume-weighted density/logit filtering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch
from torch import nn

from hgto.fem.mesh.q4 import Q4Mesh


DeviceLike = Optional[Union[str, torch.device]]


class DensityFilter(nn.Module):
    """Row-normalized, element-volume-weighted spatial hat filter.

    The expensive neighbor search is performed once at construction.  ``F``
    is retained as a coalesced torch sparse COO buffer, while ``weight_matrix``
    contains the unnormalized SciPy CSR hat weights.  The latter is useful for
    the classic top88 sensitivity filter, whose normalization is different.

    Plain boundary truncation makes the normalized operator non-symmetric on
    finite grids even when the underlying hat weights are symmetric.  This is
    a direct consequence of local row normalization.
    """

    def __init__(
        self,
        mesh: Q4Mesh,
        rmin: float,
        device: DeviceLike = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if float(rmin) <= 0.0:
            raise ValueError("rmin must be positive")
        if dtype != torch.float64:
            raise ValueError("DensityFilter requires torch.float64")

        self.rmin = float(rmin)
        self.n_elements = int(mesh.n_elements)
        centroids = np.asarray(mesh.element_centroids(), dtype=np.float64)
        volumes = np.asarray(mesh.element_volumes(), dtype=np.float64)
        if volumes.shape != (self.n_elements,) or np.any(volumes <= 0.0):
            raise ValueError("mesh element volumes must be strictly positive")

        rows = []
        cols = []
        data = []
        tree = cKDTree(centroids)
        neighborhoods = tree.query_ball_point(centroids, self.rmin)
        for row, neighbors in enumerate(neighborhoods):
            for col in sorted(int(value) for value in neighbors):
                distance = float(np.linalg.norm(centroids[row] - centroids[col]))
                weight = max(0.0, self.rmin - distance)
                if weight > 0.0:
                    rows.append(row)
                    cols.append(col)
                    data.append(weight)

        weights = sparse.coo_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(self.n_elements, self.n_elements),
            dtype=np.float64,
        ).tocsr()
        weights.sum_duplicates()
        weights.sort_indices()
        if np.any(np.asarray(weights.diagonal()) <= 0.0):
            raise RuntimeError("every filter row must contain its positive self weight")

        # the neighbor volume belongs in the numerator and row sum.
        weighted = weights.multiply(volumes[None, :]).tocsr()
        denominators = np.asarray(weighted.sum(axis=1), dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0.0):
            raise RuntimeError("filter row normalizers must be finite and positive")

        normalized = weighted.copy()
        for row in range(self.n_elements):
            start = normalized.indptr[row]
            end = normalized.indptr[row + 1]
            values = normalized.data[start:end] / denominators[row]
            # Make the floating row sum exactly one in the stored order.  This
            # preserves constants exactly under sparse COO matmul.
            if values.size > 1:
                values[-1] = 1.0 - float(np.sum(values[:-1], dtype=np.float64))
            else:
                values[0] = 1.0
            normalized.data[start:end] = values
        normalized.sort_indices()

        coo = normalized.tocoo()
        indices = torch.as_tensor(
            np.stack((coo.row, coo.col), axis=0), dtype=torch.long, device=device
        )
        values_t = torch.as_tensor(coo.data, dtype=dtype, device=device)
        with torch.sparse.check_sparse_tensor_invariants():
            matrix = torch.sparse_coo_tensor(
                indices,
                values_t,
                size=(self.n_elements, self.n_elements),
                dtype=dtype,
                device=values_t.device,
            ).coalesce()

        self.weight_matrix = weights
        self.normalized_matrix = normalized
        self.centroids = centroids
        self.element_volumes = volumes
        self.register_buffer("F", matrix)
        self.double()

    @property
    def matrix(self) -> torch.Tensor:
        """Alias for the stored sparse COO operator."""
        return self.F

    def _check_vector(self, value: torch.Tensor, name: str) -> None:
        if value.shape != (self.n_elements,):
            raise ValueError("{} must have shape (Ne,)".format(name))
        if value.dtype != self.F.dtype or value.device != self.F.device:
            raise TypeError("{} must match the filter dtype and device".format(name))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``F @ x`` with torch sparse autograd support."""
        self._check_vector(x, "x")
        return torch.sparse.mm(self.F, x.unsqueeze(1)).squeeze(1)

    def apply_transpose(self, g: torch.Tensor) -> torch.Tensor:
        """Apply the explicit transpose used by filter VJPs."""
        self._check_vector(g, "g")
        transpose = self.F.transpose(0, 1).coalesce()
        return torch.sparse.mm(transpose, g.unsqueeze(1)).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)


def build_density_filter(
    mesh: Q4Mesh,
    rmin: float,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float64,
) -> DensityFilter:
    """Functional constructor retained for config-driven callers."""
    return DensityFilter(mesh=mesh, rmin=rmin, device=device, dtype=dtype)
