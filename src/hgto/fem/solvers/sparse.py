"""CPU sparse direct solve with optional Pardiso and a SciPy fallback."""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

_FORCE = os.environ.get("HGTO_SPARSE_SOLVER", "").lower()

try:
    if _FORCE == "scipy":
        raise ImportError("forced scipy via HGTO_SPARSE_SOLVER")
    import pypardiso as _pardiso

    _BACKEND = "pardiso"
except ImportError:  # pragma: no cover
    _pardiso = None
    _BACKEND = "scipy"


def solver_name() -> str:
    return _BACKEND


def solve_spd(K_ff: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve the reduced SPD system K_ff u = rhs."""
    if _BACKEND == "pardiso":
        return _pardiso.spsolve(K_ff, rhs)
    return sparse_linalg.spsolve(K_ff, rhs)
