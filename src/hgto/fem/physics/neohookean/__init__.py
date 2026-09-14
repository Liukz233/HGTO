"""Two-dimensional Neo-Hookean constitutive kernels, Newton solver and adjoint.

Wang energy interpolation regularizes low-density elements. The default
small-strain calibration matches plane stress; finite-strain behavior
is the stated two-dimensional compressible model."""

from hgto.fem.physics.neohookean.newton import (  # noqa: F401
    minres_callback,
    nh_lame_parameters,
    pcg_callback,
    solve_nh_state,
    solve_reduced_tangent,
    wang2014_energy_at,
    wang2014_internal_force_at,
    wang2014_tangent_at,
)
from hgto.fem.physics.neohookean.constitutive import (  # noqa: F401
    deformation_gradient,
    deformation_jacobian,
    inverse_transpose_2x2,
    linear_energy_gp,
    linear_first_piola,
    linear_tangent,
    nh_energy_gp,
    nh_first_piola,
    nh_internal_force,
    nh_tangent,
    nh_tangent_diagonal,
    nh_tangent_matvec,
    wang2014_energy_gp,
    wang2014_energy_rho_derivative,
    wang2014_first_piola,
    wang2014_first_piola_rho_derivative,
    wang2014_gamma,
    wang2014_tangent,
)
