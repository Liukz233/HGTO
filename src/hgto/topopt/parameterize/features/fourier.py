"""Seeded random Fourier features for unit-bbox-normalized coordinates."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Map coordinates to seeded sinusoidal features.

    Callers must normalize coordinates to the unit bounding box first (see
    :func:`hgto.topopt.parameterize.features.static.unit_bbox_centroids`).  The frequency matrix
    is drawn once from ``N(0, sigma**2)`` with a private CPU generator so the
    draw depends only on ``seed``, and is registered as buffer ``B`` so
    checkpoints capture the experiment-specific sample.

    Args:
        n_freq: Number of sampled frequency vectors.
        sigma: Standard deviation of the normal frequency distribution.
        seed: Seed used only for this module's frequency draw.
        dim: Coordinate dimension, two for the Q4 design graph.
    """

    def __init__(self, n_freq: int, sigma: float, seed: int, dim: int = 2):
        super().__init__()
        if n_freq <= 0:
            raise ValueError("n_freq must be positive")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.n_freq = int(n_freq)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.dim = int(dim)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        frequencies = torch.randn(self.n_freq, self.dim, generator=generator, dtype=torch.float64)
        self.register_buffer("B", frequencies * self.sigma)
        self.double()  # fp64 by construction.

    @property
    def out_features(self) -> int:
        """Number of features produced for each coordinate."""
        return 2 * self.n_freq

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Return ``[sin(2*pi*B*c), cos(2*pi*B*c)]`` along the last axis."""
        if coords.ndim < 1 or coords.shape[-1] != self.dim:
            raise ValueError(
                "coords must have final dimension {}, got {}".format(self.dim, tuple(coords.shape))
            )
        projection = (2.0 * math.pi) * torch.matmul(
            coords.to(dtype=self.B.dtype), self.B.transpose(0, 1)
        )
        return torch.cat((torch.sin(projection), torch.cos(projection)), dim=-1)
