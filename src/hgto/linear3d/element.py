"""Independent NumPy Hex8 quadrature and isotropic element stiffness."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

# Reference cube [-1, 1]^3: corner a has local coords (xi_a, eta_a, zeta_a)
# and shape function N_a = (1 + xi*xi_a)(1 + eta*eta_a)(1 + zeta*zeta_a)/8.
# Bottom face CCW from (-1, -1) at zeta = -1, then the same order at +1.
_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_GAUSS_ABSCISSA = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = _GAUSS_ABSCISSA * _CORNERS
GAUSS_WEIGHTS = np.ones(8, dtype=np.float64)


def shape_functions(xi_eta_zeta: np.ndarray) -> np.ndarray:
    """Trilinear hex8 shape values, shape ``(n_points, 8)``, binding node order."""
    points = np.atleast_2d(np.asarray(xi_eta_zeta, dtype=np.float64))
    xi = points[:, 0:1]
    eta = points[:, 1:2]
    zeta = points[:, 2:3]
    return 0.125 * (
        (1.0 + xi * _CORNERS[None, :, 0])
        * (1.0 + eta * _CORNERS[None, :, 1])
        * (1.0 + zeta * _CORNERS[None, :, 2])
    )


def shape_gradients(xi_eta_zeta: np.ndarray) -> np.ndarray:
    """Reference gradients ``dN/d(xi, eta, zeta)``, shape ``(n_points, 8, 3)``."""
    points = np.atleast_2d(np.asarray(xi_eta_zeta, dtype=np.float64))
    xi = points[:, 0:1]
    eta = points[:, 1:2]
    zeta = points[:, 2:3]
    xi_a = _CORNERS[None, :, 0]
    eta_a = _CORNERS[None, :, 1]
    zeta_a = _CORNERS[None, :, 2]
    d_dxi = 0.125 * xi_a * (1.0 + eta * eta_a) * (1.0 + zeta * zeta_a)
    d_deta = 0.125 * eta_a * (1.0 + xi * xi_a) * (1.0 + zeta * zeta_a)
    d_dzeta = 0.125 * zeta_a * (1.0 + xi * xi_a) * (1.0 + eta * eta_a)
    return np.stack((d_dxi, d_deta, d_dzeta), axis=2)


# Reference-element tables evaluated once at the Gauss points.
_N_AT_GAUSS = shape_functions(GAUSS_POINTS)
_DN_AT_GAUSS = shape_gradients(GAUSS_POINTS)


def isotropic_C_3d(E: float, nu: float) -> np.ndarray:
    """Full 3D isotropic constitutive matrix, Voigt ``[xx, yy, zz, yz, xz, xy]``.

    Engineering shear convention: the shear rows multiply ``gamma = 2*eps``,
    so the diagonal shear entries are ``mu`` (not ``2*mu``).
    """
    E_value = float(E)
    nu_value = float(nu)
    if E_value <= 0.0:
        raise ValueError("E must be positive")
    if not -1.0 < nu_value < 0.5:
        raise ValueError("3D nu must lie in (-1, 0.5)")
    lam = E_value * nu_value / ((1.0 + nu_value) * (1.0 - 2.0 * nu_value))
    mu = E_value / (2.0 * (1.0 + nu_value))
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _kinematics_at_point(coords8: np.ndarray, dN_dxi: np.ndarray) -> Tuple[np.ndarray, float]:
    """B-matrix (6, 24) and Jacobian determinant at one reference point."""
    coords_array = np.asarray(coords8, dtype=np.float64)
    if coords_array.shape != (8, 3):
        raise ValueError("coords8 must have shape (8, 3)")
    jacobian = coords_array.T.dot(dN_dxi)  # (3, 3): d(x, y, z)/d(xi, eta, zeta)
    det_jacobian = float(np.linalg.det(jacobian))
    if not np.isfinite(det_jacobian) or det_jacobian <= 0.0:
        raise ValueError("hex8 element must have a positive finite Jacobian")
    dN_dx = dN_dxi.dot(np.linalg.inv(jacobian))  # (8, 3): dN/d(x, y, z)
    B = np.zeros((6, 24), dtype=np.float64)
    B[0, 0::3] = dN_dx[:, 0]  # eps_xx
    B[1, 1::3] = dN_dx[:, 1]  # eps_yy
    B[2, 2::3] = dN_dx[:, 2]  # eps_zz
    B[3, 1::3] = dN_dx[:, 2]  # gamma_yz = duy/dz + duz/dy
    B[3, 2::3] = dN_dx[:, 1]
    B[4, 0::3] = dN_dx[:, 2]  # gamma_xz = dux/dz + duz/dx
    B[4, 2::3] = dN_dx[:, 0]
    B[5, 0::3] = dN_dx[:, 1]  # gamma_xy = dux/dy + duy/dx
    B[5, 1::3] = dN_dx[:, 0]
    return B, det_jacobian


def element_stiffness(coords8: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Integrate the ``24 x 24`` stiffness of one isoparametric hex8 element."""
    C = isotropic_C_3d(E, nu)
    Ke = np.zeros((24, 24), dtype=np.float64)
    for weight, dN_dxi in zip(GAUSS_WEIGHTS, _DN_AT_GAUSS):
        B, det_jacobian = _kinematics_at_point(coords8, dN_dxi)
        Ke += (weight * det_jacobian) * B.T.dot(C).dot(B)
    return 0.5 * (Ke + Ke.T)
