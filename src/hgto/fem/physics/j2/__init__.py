"""Small-strain plane-strain J2 constitutive response and incremental solver."""

from hgto.fem.physics.j2.constitutive import (
    j2_return_mapping,
    plane_strain_to_tensor,
    simp_scaled_j2_material,
    tensor_stress_to_plane_voigt,
)
from hgto.fem.physics.j2.solver import solve_j2_history

__all__ = [
    "j2_return_mapping",
    "plane_strain_to_tensor",
    "simp_scaled_j2_material",
    "tensor_stress_to_plane_voigt",
    "solve_j2_history",
]
