"""Geometric multigrid preconditioning for matrix-free elasticity.

Bilinear or trilinear index-space transfers form element-local Galerkin
coarse operators. Chebyshev smoothing and symmetric V-cycles provide the
preconditioner for CG. Fixed geometry tables are cached by grid, device and
dtype. Optional semi-coarsening continues on even axes of rectangular grids.
The coarsest system uses a Jacobi-scaled float32 or float64 Cholesky factor;
float32 factorization failures fall back to float64. The outer state solve
and its recomputed residual remain float64."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import torch

from hgto.fem.solvers.linear import SolveFailure

Matvec = Callable[[torch.Tensor], torch.Tensor]

#: Stop coarsening once the coarsest grid has at most this many DOFs.
#: Shared by 2D and 3D. 3D level-chain sanity check for the production-scale
#: mesh 48x24x16 (62,475 fine dof): 48x24x16 -> 24x12x8 (8,775 dof, still
#: > 1024) -> 12x6x4 (1,365 dof, still > 1024) -> 6x3x2 (7*4*3 = 84 nodes
#: = 252 dof <= 1024, and nely = 3 is odd so no further halving anyway) —
#: the coarsest dense Cholesky stays a sane 252x252 factorization.
COARSEST_MAX_DOF = 1024
#: Chebyshev smoother degree (pre == post, symmetry requirement).
CHEBYSHEV_DEGREE = 3
#: Chebyshev lower-bound fraction of the upper spectral bound.
LAMBDA_RATIO = 30.0
#: Accepted ``coarsest_dtype`` spellings -> torch dtype.
COARSEST_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    torch.float64: torch.float64,
    torch.float32: torch.float32,
}
#: Cache size of the memoised rho-independent tables (per distinct
#: (extents, factors, dtype, device) key; a hierarchy uses ~3 keys per level).
_TABLE_CACHE_SIZE = 256

#: Q4 local-corner offsets (BL, BR, TR, TL — the binding CCW order).  The
#: SAME tuple doubles as the child-cell order of a 2x2 agglomerate so that
#: ``transfer_tables[c]`` and the ``_coarsen_2d`` child loop stay aligned by
#: construction.
_QUAD_OFFSETS = ((0, 0), (1, 0), (1, 1), (0, 1))

#: Hex8 local-corner offsets — bottom face CCW then top face (the binding
#: local-node order of the mesh numbering conventions, numbering_3d).  The SAME
#: tuple doubles as the child-cell order of a 2x2x2 agglomerate so that
#: ``transfer_tables[c]`` and the ``_coarsen_3d`` child loop stay aligned by
#: construction (the 2D code aligns its two inline tuples the same way).
_HEX_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)


def resolve_coarsest_dtype(value: str | torch.dtype) -> torch.dtype:
    """``"float64" | "float32" | torch.float64 | torch.float32`` -> dtype."""
    try:
        return COARSEST_DTYPES[value]
    except (KeyError, TypeError):
        raise ValueError(
            f"coarsest_dtype must be 'float64'/'float32' (or the torch dtype), got {value!r}"
        ) from None


def coarsening_factors(extents: tuple[int, ...], semi_coarsen: bool) -> tuple[int, ...] | None:
    """Per-axis agglomeration factors of one coarsening step, or ``None``.

    The binding rule (``semi_coarsen=False``): every extent even and >= 2
    -> ``(2, 2[, 2])``, else no coarsening.  With ``semi_coarsen`` the axes
    that are still even keep factor 2 and the odd ones factor 1 (identity
    transfer along that axis); ``None`` once every axis is odd.
    """
    halvable = tuple(n % 2 == 0 and n >= 2 for n in extents)
    if all(halvable):
        return tuple(2 for _ in extents)
    if not semi_coarsen or not any(halvable):
        return None
    return tuple(2 if h else 1 for h in halvable)


def _child_offsets(factors: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Child cells of one agglomerate in the binding corner order, filtered
    to the axes that are actually coarsened (factor 2)."""
    base = _HEX_OFFSETS if len(factors) == 3 else _QUAD_OFFSETS
    return tuple(offset for offset in base if all(o < f for o, f in zip(offset, factors)))


# ---------------------------------------------------------------------------
# rho-independent tables: built once per key, memoised, treated read-only
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _structured_econn(nelx: int, nely: int, device: torch.device) -> torch.Tensor:
    """Q4 connectivity of the binding structured grid (BL,BR,TR,TL CCW)."""
    ex = torch.arange(nelx, dtype=torch.long, device=device)
    ey = torch.arange(nely, dtype=torch.long, device=device)
    gy, gx = torch.meshgrid(ey, ex, indexing="ij")
    n0 = gy.reshape(-1) * (nelx + 1) + gx.reshape(-1)
    return torch.stack([n0, n0 + 1, n0 + nelx + 2, n0 + nelx + 1], dim=1)


def _element_dofs(econn: torch.Tensor) -> torch.Tensor:
    """(Ne, 8) global DOF ids in the (node-major, x-then-y local order)."""
    dofs = econn.new_empty((econn.shape[0], 8))
    dofs[:, 0::2] = 2 * econn
    dofs[:, 1::2] = 2 * econn + 1
    return dofs


def _transfer_tables_host(factors: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """(n_children, n_local, n_local) child-from-parent transfer tables
    ``T_c`` built on the host (Python scalar writes into a CPU tensor).

    Child c of a coarse element occupies the fine-node offset box
    ``prod_i [c_i, c_i + 1]`` of the coarse element's fine-node patch; a fine
    node at offset g carries the coarse-corner multilinear weights evaluated
    at ``xi_i = g_i / f_i`` (``f_i`` the per-axis factor: 2 = halved, 1 =
    identity along that axis) — the 2D bilinear / 3D trilinear hat family
    written on [0, 1].  Local node order is the binding CCW (bottom-then-top)
    order; DOFs interleave per node (componentwise map).
    """
    n_dim = len(factors)
    corners = _HEX_OFFSETS if n_dim == 3 else _QUAD_OFFSETS
    children = _child_offsets(factors)
    n_local = n_dim * len(corners)
    tables = torch.zeros((len(children), n_local, n_local), dtype=dtype)
    for c, child in enumerate(children):
        for a, node in enumerate(corners):  # child-local node a
            coords = [(child[i] + node[i]) / factors[i] for i in range(n_dim)]
            for b, corner in enumerate(corners):  # coarse-corner node b
                weight = 1.0
                for i in range(n_dim):
                    weight = weight * (coords[i] if corner[i] else 1.0 - coords[i])
                for d in range(n_dim):
                    tables[c, n_dim * a + d, n_dim * b + d] = weight
    return tables


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _transfer_tables(
    factors: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Memoised device copy of :func:`_transfer_tables_host` (one H2D copy)."""
    return _transfer_tables_host(factors, dtype).to(device=device)


def _child_transfer_tables(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """(4, 8, 8) bilinear child-from-parent transfer tables ``T_c`` (2x2)."""
    return _transfer_tables((2, 2), dtype, device)


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _child_ids(
    nelx_c: int, nely_c: int, factors: tuple[int, int], device: torch.device
) -> torch.Tensor:
    """(n_children, Ne_c) fine element ids of every coarse element's children."""
    fx, fy = factors
    nelx_f = fx * nelx_c
    ex = torch.arange(nelx_c, dtype=torch.long, device=device)
    ey = torch.arange(nely_c, dtype=torch.long, device=device)
    gy, gx = torch.meshgrid(ey, ex, indexing="ij")
    base_x = fx * gx.reshape(-1)
    base_y = fy * gy.reshape(-1)
    return torch.stack(
        [(base_y + cy) * nelx_f + (base_x + cx) for cx, cy in _child_offsets(factors)],
        dim=0,
    )


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _prolongation_tables(
    nelx_c: int,
    nely_c: int,
    device: torch.device,
    dtype: torch.dtype,
    factors: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodal bilinear prolongation as gather tables.

    Returns ``(indices, weights)`` of shape (Nn_f, 4): fine node values are
    ``u_f[n] = sum_k weights[n, k] * u_c[indices[n, k]]``.  Entries beyond a
    node's true parent count are padded with (index 0, weight 0).  The same
    tables scatter-transpose into the restriction ``P^T``.  An axis with
    factor 1 maps every fine node onto its own coarse node (unit weight).
    """
    fx, fy = factors
    nelx_f, nely_f = fx * nelx_c, fy * nely_c
    ix = torch.arange(nelx_f + 1, dtype=torch.long, device=device)
    iy = torch.arange(nely_f + 1, dtype=torch.long, device=device)
    gy, gx = torch.meshgrid(iy, ix, indexing="ij")
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)
    x0 = gx // fx
    y0 = gy // fy
    odd_x = (gx % fx).to(dtype)
    odd_y = (gy % fy).to(dtype)
    # parent corners (x0,y0), (x0+1,y0), (x0,y0+1), (x0+1,y0+1) clamped
    x1 = torch.clamp(x0 + 1, max=nelx_c)
    y1 = torch.clamp(y0 + 1, max=nely_c)
    stride = nelx_c + 1
    indices = torch.stack(
        [
            y0 * stride + x0,
            y0 * stride + x1,
            y1 * stride + x0,
            y1 * stride + x1,
        ],
        dim=1,
    )
    wx1 = 0.5 * odd_x  # weight on the +x parent (0 for even, 1/2 for odd)
    wx0 = 1.0 - wx1
    wy1 = 0.5 * odd_y
    wy0 = 1.0 - wy1
    weights = torch.stack([wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1], dim=1)
    return indices, weights


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _structured_econn_3d(nelx: int, nely: int, nelz: int, device: torch.device) -> torch.Tensor:
    """Hex8 connectivity of the binding structured grid (bottom CCW, then top)."""
    nxp = nelx + 1
    layer = nxp * (nely + 1)
    ex = torch.arange(nelx, dtype=torch.long, device=device)
    ey = torch.arange(nely, dtype=torch.long, device=device)
    ez = torch.arange(nelz, dtype=torch.long, device=device)
    gz, gy, gx = torch.meshgrid(ez, ey, ex, indexing="ij")
    n0 = gz.reshape(-1) * layer + gy.reshape(-1) * nxp + gx.reshape(-1)
    bottom = torch.stack([n0, n0 + 1, n0 + nxp + 1, n0 + nxp], dim=1)
    return torch.cat([bottom, bottom + layer], dim=1)


def _element_dofs_3d(econn: torch.Tensor) -> torch.Tensor:
    """(Ne, 24) global DOF ids in the (node-major, x-then-y-then-z) local order."""
    dofs = econn.new_empty((econn.shape[0], 24))
    dofs[:, 0::3] = 3 * econn
    dofs[:, 1::3] = 3 * econn + 1
    dofs[:, 2::3] = 3 * econn + 2
    return dofs


def _child_transfer_tables_3d(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """(8, 24, 24) trilinear child-from-parent transfer tables ``T_c`` (2x2x2)."""
    return _transfer_tables((2, 2, 2), dtype, device)


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _child_ids_3d(
    nelx_c: int,
    nely_c: int,
    nelz_c: int,
    factors: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """(n_children, Ne_c) fine element ids of every coarse element's children."""
    fx, fy, fz = factors
    nelx_f, nely_f = fx * nelx_c, fy * nely_c
    ex = torch.arange(nelx_c, dtype=torch.long, device=device)
    ey = torch.arange(nely_c, dtype=torch.long, device=device)
    ez = torch.arange(nelz_c, dtype=torch.long, device=device)
    gz, gy, gx = torch.meshgrid(ez, ey, ex, indexing="ij")
    base_x = fx * gx.reshape(-1)
    base_y = fy * gy.reshape(-1)
    base_z = fz * gz.reshape(-1)
    return torch.stack(
        [
            (base_z + cz) * (nelx_f * nely_f) + (base_y + cy) * nelx_f + (base_x + cx)
            for cx, cy, cz in _child_offsets(factors)
        ],
        dim=0,
    )


@functools.lru_cache(maxsize=_TABLE_CACHE_SIZE)
def _prolongation_tables_3d(
    nelx_c: int,
    nely_c: int,
    nelz_c: int,
    device: torch.device,
    dtype: torch.dtype,
    factors: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodal trilinear prolongation as gather tables of shape (Nn_f, 8).

    Same contract as the 2D ``_prolongation_tables``: fine node values are
    ``u_f[n] = sum_k weights[n, k] * u_c[indices[n, k]]``.  Fine nodes with
    all-even indices coincide with a coarse node (single unit weight); each
    odd axis splits the weight 1/2-1/2 between the two flanking parents.
    Clamped +1 parents carry weight zero, so padding never contributes.  The
    same tables scatter-transpose into the restriction ``P^T``.  An axis
    with factor 1 is the identity (every fine node is its own parent).
    """
    fx, fy, fz = factors
    nelx_f, nely_f, nelz_f = fx * nelx_c, fy * nely_c, fz * nelz_c
    ix = torch.arange(nelx_f + 1, dtype=torch.long, device=device)
    iy = torch.arange(nely_f + 1, dtype=torch.long, device=device)
    iz = torch.arange(nelz_f + 1, dtype=torch.long, device=device)
    gz, gy, gx = torch.meshgrid(iz, iy, ix, indexing="ij")
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)
    gz = gz.reshape(-1)
    x0 = gx // fx
    y0 = gy // fy
    z0 = gz // fz
    x1 = torch.clamp(x0 + 1, max=nelx_c)
    y1 = torch.clamp(y0 + 1, max=nely_c)
    z1 = torch.clamp(z0 + 1, max=nelz_c)
    stride_x = nelx_c + 1
    stride_z = (nelx_c + 1) * (nely_c + 1)
    # parent order: x-fastest over (x0,x1), then (y0,y1), then (z0,z1)
    indices = torch.stack(
        [z * stride_z + y * stride_x + x for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)],
        dim=1,
    )
    wx1 = 0.5 * (gx % fx).to(dtype)  # weight on the +x parent
    wx0 = 1.0 - wx1
    wy1 = 0.5 * (gy % fy).to(dtype)
    wy0 = 1.0 - wy1
    wz1 = 0.5 * (gz % fz).to(dtype)
    wz0 = 1.0 - wz1
    weights = torch.stack(
        [wz * wy * wx for wz in (wz0, wz1) for wy in (wy0, wy1) for wx in (wx0, wx1)],
        dim=1,
    )
    return indices, weights


def clear_table_cache() -> None:
    """Drop every memoised rho-independent table (tests / device teardown)."""
    for fn in (
        _structured_econn,
        _transfer_tables,
        _child_ids,
        _prolongation_tables,
        _structured_econn_3d,
        _child_ids_3d,
        _prolongation_tables_3d,
    ):
        fn.cache_clear()


class MGHierarchy:
    """Per-solve geometric hierarchy; ``apply`` is one symmetric V-cycle.

    Built from an operator + density (the element Youngs moduli fix the
    level-0 matrices).  All level vectors are FULL-length masked DOF vectors
    (fixed entries pinned to zero), mirroring the operator's ``_masked``
    convention.

    Coarse-grid options:

    ``coarsest_dtype``
        ``torch.float64`` (default): exact fp64 Cholesky of the coarsest
        free block.  ``torch.float32``: fp32 Cholesky of the Jacobi-scaled
        block ``D^-1/2 A_c D^-1/2``, fp64 fallback when potrf fails
        (``coarsest_fp32_fallback`` / module counter
        :func:`coarsest_fp32_fallback_count`); the coarsest solve then runs
        in fp32 with the scaling undone in fp64.
    ``semi_coarsen``
        ``False`` (default): coarsen only while every extent is even.
        ``True``: when an extent turns odd keep halving the even axes
        (per-axis factors, identity transfer along the odd axis).

    ``stamp()`` returns the recipe-level description of the built chain.
    """

    def __init__(
        self,
        Ke0: torch.Tensor,
        E_e: torch.Tensor,
        econn: torch.Tensor,
        free_dof_mask: torch.Tensor,
        nelx: int,
        nely: int,
        nelz: int | None = None,
        coarsest_max_dof: int = COARSEST_MAX_DOF,
        element_matrices: torch.Tensor | None = None,
        coarsest_dtype: str | torch.dtype = torch.float64,
        semi_coarsen: bool = False,
    ) -> None:
        # ``element_matrices`` provides explicit per-element
        # level-0 matrices, e.g. the consistent elastoplastic tangent
        # ``sum_g w_g B^T D_g B`` — everything downstream (masking, Galerkin
        # coarsening, smoother bounds, coarsest Cholesky) is already generic
        # over per-element matrices; only the scalar-modulus product below
        # assumed otherwise.  When given, ``Ke0``/``E_e`` are IGNORED for the
        # level-0 matrices; the default ``None`` path is byte-identical.
        device = Ke0.device
        dtype = Ke0.dtype
        self.coarsest_dtype = resolve_coarsest_dtype(coarsest_dtype)
        self.semi_coarsen = bool(semi_coarsen)
        # Dimension dispatch: ``nelz is None`` selects the
        # 2D bilinear path; an integer nelz selects the
        # hex8 trilinear path.  The 2D table functions are byte-identical.
        self.n_dim = 2 if nelz is None else 3
        n_nodes0 = free_dof_mask.numel() // self.n_dim
        if self.n_dim == 2:
            expected_nodes = (nelx + 1) * (nely + 1)
            expected_elements = nelx * nely
            grid_repr = f"nelx={nelx}, nely={nely}"
        else:
            expected_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
            expected_elements = nelx * nely * nelz
            grid_repr = f"nelx={nelx}, nely={nely}, nelz={nelz}"
        if n_nodes0 != expected_nodes or econn.shape[0] != expected_elements:
            raise ValueError(
                "MGCG requires the binding structured grid "
                f"(got {econn.shape[0]} elements for {grid_repr})"
            )

        # ---- level 0 masked element matrices (assembles exactly M K M) ----
        if element_matrices is not None:
            expected_local = 8 if self.n_dim == 2 else 24
            if element_matrices.shape != (econn.shape[0], expected_local, expected_local):
                raise ValueError(
                    f"element_matrices must have shape (Ne, {expected_local}, {expected_local})"
                )
            A0 = element_matrices.to(device=device, dtype=dtype)
        else:
            A0 = Ke0 * E_e[:, None, None]
        edofs0 = _element_dofs(econn) if self.n_dim == 2 else _element_dofs_3d(econn)
        local_free0 = free_dof_mask[edofs0].to(dtype)  # (Ne, 8 or 24)
        A0 = A0 * local_free0[:, :, None] * local_free0[:, None, :]

        # Level-0 full-agglomerate tables kept under the historical name
        # (read-only, memoised; per-level tables live in level["transfer"]).
        self.transfer_tables = (
            _child_transfer_tables(dtype, device)
            if self.n_dim == 2
            else _child_transfer_tables_3d(dtype, device)
        )
        self.levels: list[dict[str, object]] = []

        # level entries hold everything the cycle needs; level 0 keeps its
        # matrices only transiently unless it is also the coarsest level.
        current = {
            "nelx": nelx,
            "nely": nely,
            "nelz": nelz,
            "econn": econn,
            "edofs": edofs0,
            "n_dof": free_dof_mask.numel(),
            "A": A0,
        }
        while True:
            factors = self._coarsening_factors(current)
            if factors is None:
                break
            # always take the first coarsening a regular grid admits; descend
            # further only while the level is too big for a dense coarsest solve
            if self.levels and current["n_dof"] <= coarsest_max_dof:
                break
            current["factors"] = factors
            self.levels.append(current)
            current = self._coarsen(current, factors, device, dtype)
        current["factors"] = None
        self.levels.append(current)
        if len(self.levels) < 2:
            mesh_repr = f"{nelx}x{nely}" if self.n_dim == 2 else f"{nelx}x{nely}x{nelz}"
            block = "2x2" if self.n_dim == 2 else "2x2x2"
            raise ValueError(
                "MGCG needs at least one coarsening step; mesh "
                f"{mesh_repr} cannot agglomerate {block} (use jacobi)"
            )

        # ---- per-level diagonals, masks, transfers, smoother bounds ----
        for index, level in enumerate(self.levels):
            A = level["A"]
            edofs = level["edofs"]
            diagonal = A.new_zeros(level["n_dof"])
            diagonal.index_add_(
                0,
                edofs.reshape(-1),
                torch.diagonal(A, dim1=1, dim2=2).reshape(-1),
            )
            free = diagonal != 0.0
            level["free"] = free
            level["diag"] = diagonal
            level["inv_diag"] = torch.where(
                free,
                1.0 / torch.where(free, diagonal, torch.ones_like(diagonal)),
                torch.zeros_like(diagonal),
            )
            if index + 1 < len(self.levels):
                coarse = self.levels[index + 1]
                factors = level["factors"]
                if self.n_dim == 2:
                    level["prolong_idx"], level["prolong_w"] = _prolongation_tables(
                        coarse["nelx"], coarse["nely"], device, dtype, factors
                    )
                else:
                    level["prolong_idx"], level["prolong_w"] = _prolongation_tables_3d(
                        coarse["nelx"],
                        coarse["nely"],
                        coarse["nelz"],
                        device,
                        dtype,
                        factors,
                    )

        # coarsest: dense assembly + Cholesky of the free-free block
        coarsest = self.levels[-1]
        n_dof_c = coarsest["n_dof"]
        dense = torch.zeros((n_dof_c, n_dof_c), dtype=dtype, device=device)
        edofs_c = coarsest["edofs"]
        n_local = edofs_c.shape[1]  # 8 (Q4) or 24 (hex8)
        rows = edofs_c[:, :, None].expand(-1, n_local, n_local).reshape(-1)
        cols = edofs_c[:, None, :].expand(-1, n_local, n_local).reshape(-1)
        dense.index_put_((rows, cols), coarsest["A"].reshape(-1), accumulate=True)
        free_c = coarsest["free"]
        free_idx = torch.nonzero(free_c, as_tuple=False).reshape(-1)
        self._coarsest_free_idx = free_idx
        if free_idx.numel() == n_dof_c:
            block = dense  # every coarse dof free: the gather would be a copy
        else:
            block = dense[free_idx][:, free_idx]
        del dense
        self.coarsest_fp32_fallback = False
        self._coarsest_chol32: torch.Tensor | None = None
        self._coarsest_scale: torch.Tensor | None = None
        self._coarsest_chol: torch.Tensor | None = None
        if self.coarsest_dtype == torch.float32:
            # Jacobi-scaled block: unit diagonal, contrast folded into the
            # off-diagonals — fp32 potrf survives SIMP contrasts ~1e6-1e9
            # that the raw block's fp32 rounding would break.
            scale = 1.0 / torch.sqrt(torch.diagonal(block))
            scaled = (block * scale[:, None]) * scale[None, :]
            factor32, info = torch.linalg.cholesky_ex(scaled.to(torch.float32))
            del scaled
            if int(info.item()) == 0:
                self._coarsest_chol32 = factor32
                self._coarsest_scale = scale
            else:
                self.coarsest_fp32_fallback = True
                _FP32_FALLBACKS["count"] += 1
        if self._coarsest_chol32 is None:
            self._coarsest_chol = torch.linalg.cholesky(block)
        del block

        # spectral bounds AFTER diagonals exist (Gershgorin per level)
        for index in range(len(self.levels) - 1):
            self.levels[index]["lambda_max"] = self._lambda_max_bound(index)

    # ---- construction helpers ----

    def _extents(self, level: dict[str, object]) -> tuple[int, ...]:
        if self.n_dim == 2:
            return (int(level["nelx"]), int(level["nely"]))
        return (int(level["nelx"]), int(level["nely"]), int(level["nelz"]))

    def _coarsening_factors(self, level: dict[str, object]) -> tuple[int, ...] | None:
        """Per-axis factors of the next agglomeration (None: stop)."""
        return coarsening_factors(self._extents(level), self.semi_coarsen)

    def _can_halve(self, level: dict[str, object]) -> bool:
        """Whether every grid dimension of ``level`` admits 2x2(x2) halving."""
        return coarsening_factors(self._extents(level), False) is not None

    def _coarsen(
        self,
        fine: dict[str, object],
        factors: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        if self.n_dim == 2:
            return self._coarsen_2d(fine, factors, device, dtype)
        return self._coarsen_3d(fine, factors, device, dtype)

    def _coarsen_2d(
        self,
        fine: dict[str, object],
        factors: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        fx, fy = factors
        nelx_c = fine["nelx"] // fx
        nely_c = fine["nely"] // fy
        econn_c = _structured_econn(nelx_c, nely_c, device)
        A_f: torch.Tensor = fine["A"]  # type: ignore[assignment]
        tables = _transfer_tables(factors, dtype, device)
        child_ids = _child_ids(nelx_c, nely_c, factors, device)
        fine["transfer"] = tables

        A_c = A_f.new_zeros((nelx_c * nely_c, 8, 8))
        for c in range(tables.shape[0]):
            T = tables[c]
            A_c += torch.einsum("ji,ejk,kl->eil", T, A_f[child_ids[c]], T)
        return {
            "nelx": nelx_c,
            "nely": nely_c,
            "nelz": None,
            "econn": econn_c,
            "edofs": _element_dofs(econn_c),
            "n_dof": 2 * (nelx_c + 1) * (nely_c + 1),
            "A": A_c,
        }

    def _coarsen_3d(
        self,
        fine: dict[str, object],
        factors: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        fx, fy, fz = factors
        nelx_c = fine["nelx"] // fx
        nely_c = fine["nely"] // fy
        nelz_c = fine["nelz"] // fz
        econn_c = _structured_econn_3d(nelx_c, nely_c, nelz_c, device)
        A_f: torch.Tensor = fine["A"]  # type: ignore[assignment]
        tables = _transfer_tables(factors, dtype, device)
        child_ids = _child_ids_3d(nelx_c, nely_c, nelz_c, factors, device)
        fine["transfer"] = tables

        A_c = A_f.new_zeros((nelx_c * nely_c * nelz_c, 24, 24))
        for c in range(tables.shape[0]):
            T = tables[c]
            A_c += torch.einsum("ji,ejk,kl->eil", T, A_f[child_ids[c]], T)
        return {
            "nelx": nelx_c,
            "nely": nely_c,
            "nelz": nelz_c,
            "econn": econn_c,
            "edofs": _element_dofs_3d(econn_c),
            "n_dof": 3 * (nelx_c + 1) * (nely_c + 1) * (nelz_c + 1),
            "A": A_c,
        }

    def _level_matvec(self, index: int, u: torch.Tensor) -> torch.Tensor:
        level = self.levels[index]
        A: torch.Tensor = level["A"]  # type: ignore[assignment]
        edofs: torch.Tensor = level["edofs"]  # type: ignore[assignment]
        element_u = u[edofs]  # (Ne, 8 or 24)
        element_f = torch.einsum("eij,ej->ei", A, element_u)
        out = torch.zeros_like(u)
        out.index_add_(0, edofs.reshape(-1), element_f.reshape(-1))
        return out

    def _lambda_max_bound(self, index: int) -> float:
        """Guaranteed upper bound on ``lambda_max(D^{-1} A)`` via Gershgorin.

        Row absolute sums scatter exactly from the stored element matrices,
        so the bound is deterministic, matvec-free and NEVER an
        underestimate — a power-iteration estimate on late-OC binarized
        SIMP fields (contrast ~1e6) can undershoot, which makes the
        Chebyshev polynomial DIVERGENT on the missed tail and silently
        destroys the preconditioner's positive definiteness (observed as
        every solve limping to the direct fallback). Overestimating only
        weakens smoothing mildly; the V-cycle's coarse levels carry the low
        end regardless.
        """
        level = self.levels[index]
        A: torch.Tensor = level["A"]  # type: ignore[assignment]
        edofs: torch.Tensor = level["edofs"]  # type: ignore[assignment]
        inv_diag: torch.Tensor = level["inv_diag"]  # type: ignore[assignment]
        free: torch.Tensor = level["free"]  # type: ignore[assignment]
        row_abs = A.new_zeros(level["n_dof"])
        row_abs.index_add_(0, edofs.reshape(-1), torch.sum(torch.abs(A), dim=2).reshape(-1))
        ratios = (row_abs * inv_diag)[free]
        if ratios.numel() == 0:
            return 1.0
        return max(float(torch.max(ratios).item()), 1.0e-12)

    # ---- description ----

    def stamp(self) -> dict[str, object]:
        """Recipe-level description of the built chain (plain Python)."""
        used = "float32" if self._coarsest_chol32 is not None else "float64"
        levels = []
        for level in self.levels:
            grid = "x".join(
                str(level[key]) for key in ("nelx", "nely", "nelz") if level.get(key) is not None
            )
            entry: dict[str, object] = {"grid": grid, "n_dof": int(level["n_dof"])}
            if level.get("factors") is not None:
                entry["factors"] = "x".join(str(f) for f in level["factors"])
            levels.append(entry)
        return {
            "mgcg_coarsest_dtype": (
                "float32" if self.coarsest_dtype == torch.float32 else "float64"
            ),
            "mgcg_coarsest_dtype_used": used,
            "mgcg_coarsest_fp32_fallback": bool(self.coarsest_fp32_fallback),
            "mgcg_semi_coarsen": bool(self.semi_coarsen),
            "mgcg_levels": levels,
            "mgcg_coarsest_dof": int(self._coarsest_free_idx.numel()),
        }

    # ---- cycle application ----

    def _smooth(self, index: int, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Chebyshev(deg) smoothing sweeps on level ``index`` (in place on x)."""
        level = self.levels[index]
        inv_diag: torch.Tensor = level["inv_diag"]  # type: ignore[assignment]
        lam_max = float(level["lambda_max"])  # Gershgorin: already an upper bound
        lam_min = lam_max / LAMBDA_RATIO
        theta = 0.5 * (lam_max + lam_min)
        delta = 0.5 * (lam_max - lam_min)
        sigma = theta / delta
        rho = 1.0 / sigma
        residual = rhs - self._level_matvec(index, x)
        d = (inv_diag * residual) / theta
        for step in range(CHEBYSHEV_DEGREE):
            x = x + d
            if step + 1 == CHEBYSHEV_DEGREE:
                break
            residual = residual - self._level_matvec(index, d)
            rho_next = 1.0 / (2.0 * sigma - rho)
            d = rho_next * rho * d + (2.0 * rho_next / delta) * (inv_diag * residual)
            rho = rho_next
        return x

    def _restrict(self, index: int, residual: torch.Tensor) -> torch.Tensor:
        """P^T scatter from level ``index`` to ``index + 1`` (masked)."""
        level = self.levels[index]
        coarse = self.levels[index + 1]
        idx: torch.Tensor = level["prolong_idx"]  # type: ignore[assignment]
        w: torch.Tensor = level["prolong_w"]  # type: ignore[assignment]
        field = residual.reshape(-1, self.n_dim)
        out = field.new_zeros((coarse["n_dof"] // self.n_dim, self.n_dim))
        contributions = w[:, :, None] * field[:, None, :]  # (Nn_f, 4 or 8, dim)
        out.index_add_(0, idx.reshape(-1), contributions.reshape(-1, self.n_dim))
        result = out.reshape(-1)
        return result * coarse["free"].to(result.dtype)

    def _prolong(self, index: int, u_coarse: torch.Tensor) -> torch.Tensor:
        """Bi/trilinear gather from level ``index + 1`` to ``index`` (masked)."""
        level = self.levels[index]
        idx: torch.Tensor = level["prolong_idx"]  # type: ignore[assignment]
        w: torch.Tensor = level["prolong_w"]  # type: ignore[assignment]
        field = u_coarse.reshape(-1, self.n_dim)
        gathered = field[idx]  # (Nn_f, 4 or 8, dim)
        fine = torch.sum(w[:, :, None] * gathered, dim=1).reshape(-1)
        return fine * self.levels[index]["free"].to(fine.dtype)

    def _coarsest_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        idx = self._coarsest_free_idx
        if self._coarsest_chol32 is not None:
            scale = self._coarsest_scale
            reduced = (rhs[idx] * scale).reshape(-1, 1)
            solution = (
                torch.cholesky_solve(reduced.to(torch.float32), self._coarsest_chol32)
                .to(rhs.dtype)
                .reshape(-1)
                * scale
            )
        else:
            reduced = rhs[idx].reshape(-1, 1)
            solution = torch.cholesky_solve(reduced, self._coarsest_chol).reshape(-1)
        out = torch.zeros_like(rhs)
        out[idx] = solution
        return out

    def _vcycle(self, index: int, rhs: torch.Tensor) -> torch.Tensor:
        if index == len(self.levels) - 1:
            return self._coarsest_solve(rhs)
        x = self._smooth(index, torch.zeros_like(rhs), rhs)
        residual = rhs - self._level_matvec(index, x)
        coarse_rhs = self._restrict(index, residual)
        coarse_x = self._vcycle(index + 1, coarse_rhs)
        x = x + self._prolong(index, coarse_x)
        return self._smooth(index, x, rhs)

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        """One symmetric V(1,1) cycle: ``z approx A^{-1} r`` (SPD map)."""
        masked = residual * self.levels[0]["free"].to(residual.dtype)
        return self._vcycle(0, masked)


#: Process-wide count of fp32 coarsest factorizations that fell back to fp64.
_FP32_FALLBACKS = {"count": 0}


def coarsest_fp32_fallback_count() -> int:
    """How many ``coarsest_dtype=float32`` builds fell back to fp64 so far."""
    return int(_FP32_FALLBACKS["count"])


def preconditioned_cg(
    matvec: Matvec,
    rhs: torch.Tensor,
    preconditioner: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    rtol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Conjugate gradients with an SPD preconditioner callable.

    The residual is recomputed from the operator at every iteration. Device
    scalars share a host transfer for convergence and finite-value checks."""
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    x = initial.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    normalizer = torch.where(rhs_norm > 0.0, rhs_norm, torch.ones_like(rhs_norm))
    residual = rhs - matvec(x)
    residual_rel = torch.linalg.vector_norm(residual) / normalizer
    if float(residual_rel.item()) <= rtol:
        return x, residual_rel, 0, True

    z = preconditioner(residual)
    direction = z.clone()
    rz = torch.dot(residual, z)
    rz_value = float(rz.item())
    if not math.isfinite(rz_value) or rz_value <= 0.0:
        raise SolveFailure(
            "MGCG preconditioner produced a non-SPD action",
            kind="mgcg",
            residual_rel=float(residual_rel.item()),
            iterations=0,
        )
    converged = False
    iterations = 0
    pending_rz: torch.Tensor | None = None  # rz_new whose check is deferred
    for iteration in range(1, max_iter + 1):
        product = matvec(direction)
        curvature = torch.dot(direction, product)
        alpha = rz / curvature
        x_candidate = x + alpha * direction
        residual_candidate = rhs - matvec(x_candidate)  # true residual (certificate)
        residual_rel_candidate = torch.linalg.vector_norm(residual_candidate) / normalizer
        if pending_rz is None:
            curvature_value, residual_rel_value = torch.stack(
                [curvature, residual_rel_candidate]
            ).tolist()
        else:
            pending_value, curvature_value, residual_rel_value = torch.stack(
                [pending_rz, curvature, residual_rel_candidate]
            ).tolist()
            # the historical post-update check of the previous iteration
            if not math.isfinite(pending_value) or rz_value == 0.0:
                break
            rz_value = pending_value
        if not math.isfinite(curvature_value) or curvature_value <= 0.0:
            break  # indefinite/broken operator: certify failure below
        x = x_candidate
        residual = residual_candidate
        residual_rel = residual_rel_candidate
        iterations = iteration
        if residual_rel_value <= rtol:
            converged = True
            break
        if iteration == max_iter:
            break  # the direction update would be dead work
        z = preconditioner(residual)
        rz_new = torch.dot(residual, z)
        beta = rz_new / rz
        direction = z + beta * direction
        pending_rz = rz_new
        rz = rz_new

    return x, residual_rel, iterations, converged
