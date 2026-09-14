"""Matrix-free conjugate-gradient and sparse direct linear solvers."""

from hgto.fem.solvers.linear import (
    SolveFailure,
    assemble_sparse_stiffness,
    direct_solve_reduced,
    pcg_callback,
)

__all__ = [
    "SolveFailure",
    "assemble_sparse_stiffness",
    "direct_solve_reduced",
    "pcg_callback",
]
