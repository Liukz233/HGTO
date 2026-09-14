"""Conventional hat density filter and optional normalized tanh projection."""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


class DensityMap:
    def __init__(self, centroids, radius):
        neighbors = cKDTree(centroids).query_ball_point(centroids, radius)
        rows = []
        cols = []
        values = []
        for i, js in enumerate(neighbors):
            js = np.asarray(sorted(js))
            w = np.maximum(0.0, radius - np.linalg.norm(centroids[i] - centroids[js], axis=1))
            rows.extend([i] * len(js))
            cols.extend(js)
            values.extend(w)
        H = sparse.csr_matrix((values, (rows, cols)), shape=(len(centroids), len(centroids)))
        self.A = sparse.diags(1 / np.asarray(H.sum(1)).ravel()) @ H

    def physical(self, x, beta=0.0):
        bar = self.A @ x
        if not beta:
            return bar, np.ones_like(bar)
        t = np.tanh(beta * (bar - 0.5))
        den = 2 * np.tanh(beta / 2)
        return (np.tanh(beta / 2) + t) / den, beta * (1 - t * t) / den

    def pullback(self, g, derivative):
        return self.A.T @ (g * derivative)


def oc_update(x, dc, dv, density_map, volume, beta, move=0.2):
    """Standard multiplicative OC with physical-volume bisection."""
    lo = np.maximum(0.0, x - move)
    hi = np.minimum(1.0, x + move)
    scale = np.sqrt(np.maximum(-dc, 0.0) / np.maximum(dv, 1e-30))
    left, right = 0.0, max(1.0, float(np.max(scale * scale)))
    for _ in range(100):
        multiplier = (left + right) / 2
        candidate = np.clip(x * scale / np.sqrt(multiplier), lo, hi)
        rho, _ = density_map.physical(candidate, beta)
        if rho.mean() > volume:
            left = multiplier
        else:
            right = multiplier
        if (right - left) / (right + left + 1e-30) < 1e-7:
            break
    return candidate, rho
