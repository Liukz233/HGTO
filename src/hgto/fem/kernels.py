"""Graph mechanics kernels for Q4 and Hex8 linear elasticity.

Geometry tables are computed once. Nodal displacements are gathered to
strain and stress at quadrature points, then force contributions are
integrated and scattered to nodes. The Gauss points form internal channels
of element incidence. Engineering shear is used in the Voigt vectors."""

from __future__ import annotations

import torch

from hgto.fem.mesh.quadrature import hex8_reference_tables, q4_reference_tables

Scalar = float | torch.Tensor

# Local-node / Gauss-point counts per spatial dimension (Q4: 4, hex8: 8).
_NODES_PER_ELEMENT = {2: 4, 3: 8}


def _scalar_like(value: Scalar, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def geometry_tables(coords_t: torch.Tensor, econn_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Physical Q4/hex8 gradient/integration tables (GNN1).

    2D returns ``dN_dx`` of shape ``(Ne, 4, 4, 2)`` (gauss, local node, xy)
    and ``detJ`` / ``wdetJ`` of shape ``(Ne, 4)``; 3D returns the hex8
    siblings ``(Ne, 8, 8, 3)`` and ``(Ne, 8)``.
    """
    if coords_t.ndim != 2 or int(coords_t.shape[1]) not in _NODES_PER_ELEMENT:
        raise ValueError("coords_t must have shape (Nn, 2) or (Nn, 3)")
    n_dim = int(coords_t.shape[1])
    nodes_per_element = _NODES_PER_ELEMENT[n_dim]
    if not coords_t.is_floating_point():
        raise TypeError("coords_t must be floating point")
    if econn_t.ndim != 2 or econn_t.shape[1] != nodes_per_element:
        raise ValueError(f"econn_t must have shape (Ne, {nodes_per_element})")
    if econn_t.dtype != torch.long:
        raise TypeError("econn_t must have dtype torch.int64")
    if econn_t.device != coords_t.device:
        raise ValueError("coords_t and econn_t must be on the same device")

    reference = q4_reference_tables() if n_dim == 2 else hex8_reference_tables()
    dN_dxi = torch.as_tensor(reference["dN_dxi"], dtype=coords_t.dtype, device=coords_t.device)
    weights = torch.as_tensor(reference["weights"], dtype=coords_t.dtype, device=coords_t.device)
    element_coords = coords_t[econn_t]

    # J_ij = sum_a X_ai dN_a/dxi_j.
    jacobian = torch.einsum("eai,gaj->egij", element_coords, dN_dxi)
    det_jacobian = torch.linalg.det(jacobian)
    if bool(torch.any(det_jacobian <= 0.0).item()):
        element_name = "Q4" if n_dim == 2 else "hex8"
        raise ValueError(f"{element_name} element has a non-positive Gauss-point Jacobian")

    inv_jacobian = torch.linalg.inv(jacobian)
    # Row form of grad_x N = J^{-T} grad_xi N.
    dN_dx = torch.einsum("gaj,egji->egai", dN_dxi, inv_jacobian)
    return {
        "dN_dx": dN_dx,
        "detJ": det_jacobian,
        "wdetJ": det_jacobian * weights.unsqueeze(0),
    }


def gauss_strains(u: torch.Tensor, econn_t: torch.Tensor, dN_dx: torch.Tensor) -> torch.Tensor:
    """Displacement-to-strain incidence operator (GNN2 kinematics).

    2D: ``u`` has load-batched shape ``(L, Nn, 2)``; returns ``(L, Ne, 4, 3)``.
    3D: ``u`` has shape ``(L, Nn, 3)``; returns ``(L, Ne, 8, 6)`` in Voigt
    order ``[xx, yy, zz, yz, xz, xy]`` with engineering shear
    ``gamma_ij = du_i/dx_j + du_j/dx_i``.
    """
    if u.ndim != 3 or int(u.shape[2]) not in _NODES_PER_ELEMENT:
        raise ValueError("u must have shape (L, Nn, 2) or (L, Nn, 3)")
    n_dim = int(u.shape[2])
    nodes_per_element = _NODES_PER_ELEMENT[n_dim]
    if econn_t.ndim != 2 or econn_t.shape[1] != nodes_per_element:
        raise ValueError(f"econn_t must have shape (Ne, {nodes_per_element})")
    if econn_t.dtype != torch.long:
        raise TypeError("econn_t must have dtype torch.int64")
    if dN_dx.shape != (econn_t.shape[0], nodes_per_element, nodes_per_element, n_dim):
        raise ValueError(
            f"dN_dx must have shape (Ne, {nodes_per_element}, {nodes_per_element}, {n_dim})"
        )
    if u.device != econn_t.device or u.device != dN_dx.device:
        raise ValueError("u, econn_t, and dN_dx must be on the same device")
    if u.dtype != dN_dx.dtype:
        raise TypeError("u and dN_dx must have the same dtype")

    element_u = u[:, econn_t, :]
    grad_u = torch.einsum("lead,egaj->legdj", element_u, dN_dx)
    if n_dim == 2:
        return torch.stack(
            [
                grad_u[..., 0, 0],
                grad_u[..., 1, 1],
                grad_u[..., 0, 1] + grad_u[..., 1, 0],
            ],
            dim=-1,
        )
    return torch.stack(
        [
            grad_u[..., 0, 0],
            grad_u[..., 1, 1],
            grad_u[..., 2, 2],
            grad_u[..., 1, 2] + grad_u[..., 2, 1],
            grad_u[..., 0, 2] + grad_u[..., 2, 0],
            grad_u[..., 0, 1] + grad_u[..., 1, 0],
        ],
        dim=-1,
    )


def plane_stress_matrix(E: Scalar, nu: Scalar, reference: torch.Tensor) -> torch.Tensor:
    """Engineering-Voigt plane-stress constitutive matrix."""
    E_t = _scalar_like(E, reference)
    nu_t = _scalar_like(nu, reference)
    one = reference.new_tensor(1.0)
    zero = reference.new_tensor(0.0)
    scale = E_t / (one - nu_t * nu_t)
    return scale * torch.stack(
        [
            torch.stack([one, nu_t, zero]),
            torch.stack([nu_t, one, zero]),
            torch.stack([zero, zero, (one - nu_t) * 0.5]),
        ]
    )


def isotropic_C_3d(E: Scalar, nu: Scalar, reference: torch.Tensor) -> torch.Tensor:
    """Full 3D isotropic constitutive matrix, Voigt ``[xx, yy, zz, yz, xz, xy]``.

    Built from the Lame parameters ``lambda`` / ``mu``. Engineering shear
    convention: the shear rows multiply ``gamma = 2*eps``, so the diagonal
    shear entries are ``mu`` (not ``2*mu``).
    """
    E_t = _scalar_like(E, reference)
    nu_t = _scalar_like(nu, reference)
    one = reference.new_tensor(1.0)
    two = reference.new_tensor(2.0)
    zero = reference.new_tensor(0.0)
    lam = E_t * nu_t / ((one + nu_t) * (one - two * nu_t))
    mu = E_t / (two * (one + nu_t))
    axial = lam + two * mu
    return torch.stack(
        [
            torch.stack([axial, lam, lam, zero, zero, zero]),
            torch.stack([lam, axial, lam, zero, zero, zero]),
            torch.stack([lam, lam, axial, zero, zero, zero]),
            torch.stack([zero, zero, zero, mu, zero, zero]),
            torch.stack([zero, zero, zero, zero, mu, zero]),
            torch.stack([zero, zero, zero, zero, zero, mu]),
        ]
    )


def _unit_constitutive(strain: torch.Tensor, nu: Scalar) -> torch.Tensor:
    """Unit-modulus C for the Voigt width of ``strain`` (3 -> 2D, 6 -> 3D)."""
    if strain.shape[-1] == 3:
        return plane_stress_matrix(1.0, nu, strain)
    return isotropic_C_3d(1.0, nu, strain)


def gauss_stress(strain: torch.Tensor, E_e: torch.Tensor, nu: Scalar) -> torch.Tensor:
    """Isotropic elasticity at every element Gauss point (GNN2 constitutive).

    2D plane stress on Voigt width 3; full 3D isotropic on Voigt width 6.
    """
    if strain.ndim != 4 or tuple(strain.shape[-2:]) not in ((4, 3), (8, 6)):
        raise ValueError("strain must have shape (L, Ne, 4, 3) or (L, Ne, 8, 6)")
    if E_e.ndim != 1 or E_e.shape[0] != strain.shape[1]:
        raise ValueError("E_e must have shape (Ne,)")
    if E_e.device != strain.device or E_e.dtype != strain.dtype:
        raise TypeError("E_e and strain must have the same dtype and device")
    C0 = _unit_constitutive(strain, nu)
    stress_unit = torch.einsum("ij,legj->legi", C0, strain)
    return stress_unit * E_e[None, :, None, None]


def internal_force(
    stress: torch.Tensor,
    econn_t: torch.Tensor,
    dN_dx: torch.Tensor,
    wdetJ: torch.Tensor,
    n_nodes: int | None = None,
) -> torch.Tensor:
    """Integrate and scatter element force messages to shared nodes (GNN3).

    ``wdetJ`` may include the physical thickness; the geometry table itself
    deliberately contains only quadrature weight times determinant.
    """
    if stress.ndim != 4 or tuple(stress.shape[-2:]) not in ((4, 3), (8, 6)):
        raise ValueError("stress must have shape (L, Ne, 4, 3) or (L, Ne, 8, 6)")
    n_dim = 2 if stress.shape[-1] == 3 else 3
    nodes_per_element = _NODES_PER_ELEMENT[n_dim]
    n_elements = econn_t.shape[0]
    if stress.shape[1] != n_elements:
        raise ValueError("stress and econn_t element counts differ")
    if dN_dx.shape != (n_elements, nodes_per_element, nodes_per_element, n_dim):
        raise ValueError(
            f"dN_dx must have shape (Ne, {nodes_per_element}, {nodes_per_element}, {n_dim})"
        )
    if wdetJ.shape != (n_elements, nodes_per_element):
        raise ValueError(f"wdetJ must have shape (Ne, {nodes_per_element})")
    if econn_t.dtype != torch.long:
        raise TypeError("econn_t must have dtype torch.int64")
    if not (stress.device == econn_t.device == dN_dx.device == wdetJ.device):
        raise ValueError("all internal_force tensors must share a device")
    if not (stress.dtype == dN_dx.dtype == wdetJ.dtype):
        raise TypeError("all floating internal_force tensors must share a dtype")

    if n_dim == 2:
        dN_dx_x = dN_dx[..., 0]
        dN_dx_y = dN_dx[..., 1]
        force_x = torch.einsum("leg,ega,eg->lea", stress[..., 0], dN_dx_x, wdetJ) + torch.einsum(
            "leg,ega,eg->lea", stress[..., 2], dN_dx_y, wdetJ
        )
        force_y = torch.einsum("leg,ega,eg->lea", stress[..., 1], dN_dx_y, wdetJ) + torch.einsum(
            "leg,ega,eg->lea", stress[..., 2], dN_dx_x, wdetJ
        )
        element_force = torch.stack([force_x, force_y], dim=-1)
    else:
        # Full symmetric stress tensor from Voigt [xx, yy, zz, yz, xz, xy],
        # then f_ad = sum_g w_g detJ_g sigma_dj dN_a/dx_j (B^T sigma).
        sigma = torch.stack(
            [
                torch.stack([stress[..., 0], stress[..., 5], stress[..., 4]], dim=-1),
                torch.stack([stress[..., 5], stress[..., 1], stress[..., 3]], dim=-1),
                torch.stack([stress[..., 4], stress[..., 3], stress[..., 2]], dim=-1),
            ],
            dim=-2,
        )
        element_force = torch.einsum("legdj,egaj,eg->lead", sigma, dN_dx, wdetJ)

    if n_nodes is None:
        n_nodes = int(torch.max(econn_t).item()) + 1
    force = stress.new_zeros((stress.shape[0], n_nodes, n_dim))
    force.index_add_(1, econn_t.reshape(-1), element_force.reshape(stress.shape[0], -1, n_dim))
    return force


def unit_strain_energy_gp(strain: torch.Tensor, nu: Scalar) -> torch.Tensor:
    """UNHALVED ``eps:C0:eps`` at each Gauss point, unit Young's modulus.

    this is the integrand compliance sensitivity consumes. The
    physical 1/2-form lives in :func:`strain_energy_gp` under its own name —
    the two must never be conflated. Identical naming contract in 2D and 3D;
    the constitutive matrix is selected by the Voigt width of ``strain``.
    """
    C0 = _unit_constitutive(strain, nu)
    stress_unit = torch.einsum("ij,legj->legi", C0, strain)
    return torch.sum(strain * stress_unit, dim=-1)


def strain_energy_gp(strain: torch.Tensor, E_e: torch.Tensor, nu: Scalar) -> torch.Tensor:
    """Physical ``0.5 * E_e * eps:C0:eps`` per Gauss point."""
    if E_e.ndim != 1 or E_e.shape[0] != strain.shape[1]:
        raise ValueError("E_e must have shape (Ne,)")
    if E_e.device != strain.device or E_e.dtype != strain.dtype:
        raise TypeError("E_e and strain must have the same dtype and device")
    return 0.5 * unit_strain_energy_gp(strain, nu) * E_e[None, :, None]
