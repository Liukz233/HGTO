"""Fourier-encoded Chebyshev graph density network with optional static caching."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import ChebConv

from hgto.topopt.parameterize.features.fourier import FourierFeatures


class ChebNetDensity(nn.Module):
    """Learnable element density from Fourier coordinates and graph convolutions.

    The network applies ChebConv, ReLU and a second ChebConv on the element
    adjacency graph. It returns logits, or sigmoid densities when requested.
    ``paper_K`` is the largest polynomial degree: PyG therefore receives
    ``K = paper_K + 1``. Optional extra input channels are concatenated after
    Fourier encoding. Static caching is available when coordinates and graph
    connectivity are fixed; weights and hidden activations remain trainable."""

    def __init__(
        self,
        n_freq: int,
        sigma: float,
        seed: int,
        hidden_dim: int = 64,
        paper_K: int = 1,
        dim: int = 2,
        extra_in: int = 0,
    ):
        super().__init__()
        assert paper_K >= 0, "paper_K must be non-negative"
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if extra_in < 0:
            raise ValueError("extra_in must be non-negative")

        self.paper_K = int(paper_K)
        self.pyg_K = self.paper_K + 1
        self.hidden_dim = int(hidden_dim)
        self.extra_in = int(extra_in)
        self.fourier = FourierFeatures(n_freq, sigma, seed, dim=dim)
        self.conv1 = ChebConv(
            self.fourier.out_features + self.extra_in, self.hidden_dim, K=self.pyg_K
        )
        self.conv2 = ChebConv(self.hidden_dim, 1, K=self.pyg_K)
        self.double()  # PyG initializes parameters in fp32.
        self._static_cache = None

    def _apply(self, fn, recurse=True):
        self._static_cache = None
        return super()._apply(fn, recurse=recurse)

    @staticmethod
    def _input_key(coords, edge_index, B):
        return tuple(
            (t.data_ptr(), t._version, tuple(t.shape), t.device) for t in (coords, edge_index, B)
        )

    @torch.no_grad()
    def prepare_static(self, coords, edge_index):
        """Cache fixed Fourier/Chebyshev input features and graph normalization.

        Only explicit fixed-coordinate callers opt in. Trainable parameters
        and hidden activations are never cached. Tensor mutation/device moves
        invalidate the cache; coordinates requiring gradients are rejected.
        The PyG propagation/summation order is retained for parity.
        """
        if coords.requires_grad or self.fourier.B.requires_grad or self.extra_in:
            raise ValueError("static cache requires fixed coordinates and no extra features")
        x = self.fourier(coords)
        edge, norm = self.conv1.__norm__(
            edge_index, x.size(0), None, self.conv1.normalization, None, dtype=x.dtype
        )
        terms = [x]
        if self.pyg_K > 1:
            terms.append(self.conv1.propagate(edge, x=x, norm=norm))
        for _ in range(2, self.pyg_K):
            terms.append(2.0 * self.conv1.propagate(edge, x=terms[-1], norm=norm) - terms[-2])
        self._static_cache = (
            self._input_key(coords, edge_index, self.fourier.B),
            terms,
            edge,
            norm,
        )

    def forward(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        return_density: bool = False,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate one logit (or density) per element centroid."""
        cache = self._static_cache
        if cache is not None:
            if coords.requires_grad or self.fourier.B.requires_grad or extra_features is not None:
                cache = None
            elif cache[0] != self._input_key(coords, edge_index, self.fourier.B):
                self.prepare_static(coords, edge_index)
                cache = self._static_cache
        if cache is not None:
            _, terms, edge, norm = cache
            x = self.conv1.lins[0](terms[0])
            for lin, term in zip(self.conv1.lins[1:], terms[1:]):
                x = x + lin(term)
            if self.conv1.bias is not None:
                x = x + self.conv1.bias
            x = F.relu(x)
            t0 = x
            out = self.conv2.lins[0](t0)
            if self.pyg_K > 1:
                t1 = self.conv2.propagate(edge, x=x, norm=norm)
                out = out + self.conv2.lins[1](t1)
                for lin in self.conv2.lins[2:]:
                    t2 = 2.0 * self.conv2.propagate(edge, x=t1, norm=norm) - t0
                    out = out + lin(t2)
                    t0, t1 = t1, t2
            if self.conv2.bias is not None:
                out = out + self.conv2.bias
            logits = out.squeeze(-1)
            return torch.sigmoid(logits) if return_density else logits
        x = self.fourier(coords)
        if self.extra_in:
            if extra_features is None or extra_features.shape != (x.shape[0], self.extra_in):
                raise ValueError(f"extra_features must have shape ({x.shape[0]}, {self.extra_in})")
            x = torch.cat([x, extra_features], dim=1)
        elif extra_features is not None:
            raise ValueError("extra_features passed to a model built with extra_in=0")
        x = F.relu(self.conv1(x, edge_index))
        logits = self.conv2(x, edge_index).squeeze(-1)
        if return_density:
            return torch.sigmoid(logits)
        return logits
