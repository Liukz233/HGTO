"""Precisely specified geometry and load cases used by the revised study."""

from .domains import make_domain_case, save_domain_case
from .spatial import make_3d_case

__all__ = ["make_domain_case", "save_domain_case", "make_3d_case"]
