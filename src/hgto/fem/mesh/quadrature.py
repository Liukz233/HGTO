"""Isoparametric shape functions and full Gauss quadrature for Q4 and Hex8.

Q4 corners are counter-clockwise from (-1, -1). Hex8 corners use matching
bottom and top faces. The integration points lie at plus/minus 1/sqrt(3)."""

from __future__ import annotations

import numpy as np

GAUSS_G = 1.0 / np.sqrt(3.0)

# (4, 2) reference coordinates of the local nodes, CCW from bottom-left.
LOCAL_NODES_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)

# (4, 2) Gauss points, CCW from (-g, -g); (4,) unit weights.
GAUSS_POINTS = LOCAL_NODES_XI * GAUSS_G
GAUSS_WEIGHTS = np.ones(4, dtype=np.float64)


def shape_functions(xi_eta: np.ndarray) -> np.ndarray:
    """Bilinear N_a(xi, eta) at arbitrary reference-square points.

    Args:
        xi_eta: (m, 2) evaluation points in the reference square.
    Returns:
        (m, 4) values, local-node order per LOCAL_NODES_XI.
    """
    xi_eta = np.atleast_2d(np.asarray(xi_eta, dtype=np.float64))
    xi, eta = xi_eta[:, 0:1], xi_eta[:, 1:2]
    xa, ea = LOCAL_NODES_XI[:, 0][None, :], LOCAL_NODES_XI[:, 1][None, :]
    return 0.25 * (1.0 + xi * xa) * (1.0 + eta * ea)


def shape_gradients(xi_eta: np.ndarray) -> np.ndarray:
    """d N_a / d(xi, eta) at arbitrary reference-square points.

    Returns:
        (m, 4, 2) gradients; last axis = (d/dxi, d/deta).
    """
    xi_eta = np.atleast_2d(np.asarray(xi_eta, dtype=np.float64))
    xi, eta = xi_eta[:, 0:1], xi_eta[:, 1:2]
    xa, ea = LOCAL_NODES_XI[:, 0][None, :], LOCAL_NODES_XI[:, 1][None, :]
    dN_dxi = 0.25 * xa * (1.0 + eta * ea)
    dN_deta = 0.25 * ea * (1.0 + xi * xa)
    return np.stack([dN_dxi, dN_deta], axis=2)


def q4_reference_tables() -> dict:
    """Packed reference-element tables at the 2x2 Gauss points.

    Returns dict with:
        N        (4 gp, 4 local nodes)
        dN_dxi   (4 gp, 4 local nodes, 2)
        points   (4, 2)
        weights  (4,)
    """
    return {
        "N": shape_functions(GAUSS_POINTS),
        "dN_dxi": shape_gradients(GAUSS_POINTS),
        "points": GAUSS_POINTS.copy(),
        "weights": GAUSS_WEIGHTS.copy(),
    }


# ---------------------------------------------------------------------------
# Hex8 (3D) reference element — BINDING orderings per quadrature_3d.
# ---------------------------------------------------------------------------

# (8, 3) reference coordinates of the local nodes: bottom face CCW from
# (-1,-1,-1), then the SAME order on the top face at zeta = +1.
LOCAL_NODES_XI_3D = np.array(
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

# (8, 3) Gauss points mirroring the local-node order; (8,) unit weights.
GAUSS_POINTS_3D = LOCAL_NODES_XI_3D * GAUSS_G
GAUSS_WEIGHTS_3D = np.ones(8, dtype=np.float64)


def shape_functions_3d(xi_eta_zeta: np.ndarray) -> np.ndarray:
    """Trilinear N_a(xi, eta, zeta) at arbitrary reference-cube points.

    Args:
        xi_eta_zeta: (m, 3) evaluation points in the reference cube.
    Returns:
        (m, 8) values, local-node order per LOCAL_NODES_XI_3D.
    """
    pts = np.atleast_2d(np.asarray(xi_eta_zeta, dtype=np.float64))
    xi, eta, zeta = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
    xa = LOCAL_NODES_XI_3D[:, 0][None, :]
    ea = LOCAL_NODES_XI_3D[:, 1][None, :]
    za = LOCAL_NODES_XI_3D[:, 2][None, :]
    return 0.125 * (1.0 + xi * xa) * (1.0 + eta * ea) * (1.0 + zeta * za)


def shape_gradients_3d(xi_eta_zeta: np.ndarray) -> np.ndarray:
    """d N_a / d(xi, eta, zeta) at arbitrary reference-cube points.

    Returns:
        (m, 8, 3) gradients; last axis = (d/dxi, d/deta, d/dzeta).
    """
    pts = np.atleast_2d(np.asarray(xi_eta_zeta, dtype=np.float64))
    xi, eta, zeta = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
    xa = LOCAL_NODES_XI_3D[:, 0][None, :]
    ea = LOCAL_NODES_XI_3D[:, 1][None, :]
    za = LOCAL_NODES_XI_3D[:, 2][None, :]
    dN_dxi = 0.125 * xa * (1.0 + eta * ea) * (1.0 + zeta * za)
    dN_deta = 0.125 * ea * (1.0 + xi * xa) * (1.0 + zeta * za)
    dN_dzeta = 0.125 * za * (1.0 + xi * xa) * (1.0 + eta * ea)
    return np.stack([dN_dxi, dN_deta, dN_dzeta], axis=2)


def hex8_reference_tables() -> dict:
    """Packed reference-element tables at the 2x2x2 Gauss points.

    Returns dict with (same key contract as q4_reference_tables):
        N        (8 gp, 8 local nodes)
        dN_dxi   (8 gp, 8 local nodes, 3)
        points   (8, 3)
        weights  (8,)
    """
    return {
        "N": shape_functions_3d(GAUSS_POINTS_3D),
        "dN_dxi": shape_gradients_3d(GAUSS_POINTS_3D),
        "points": GAUSS_POINTS_3D.copy(),
        "weights": GAUSS_WEIGHTS_3D.copy(),
    }
