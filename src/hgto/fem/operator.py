"""Matrix-free graph mechanics for Q4 plane-stress and Hex8 linear elasticity.

Element incidence gathers nodal states, local kernels evaluate constitutive
responses, and scatter operations assemble force messages. Load and state
fields have shape (n_loads, n_nodes, n_dimensions). Homogeneous prescribed
DOFs are removed from the state system. Solves check the recomputed free-DOF
residual and return float64 displacement, energy and sensitivity data."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from hgto.fem.kernels import (
    gauss_strains,
    gauss_stress,
    geometry_tables,
    internal_force,
    isotropic_C_3d,
    plane_stress_matrix,
    strain_energy_gp,
    unit_strain_energy_gp,
)
from hgto.fem.solvers.linear import (
    SolveFailure,
    assemble_sparse_stiffness,
    direct_solve_reduced,
    pcg_callback,
)
from hgto.fem.state import MechanicsState
from hgto.fem.mesh.hex8 import Hex8Mesh
from hgto.fem.mesh.q4 import Q4Mesh

DeviceLike = str | torch.device | None


class MechanicsOperator(nn.Module):
    """White-box Q4/hex8 isotropic elastic operator with certified Jacobi-PCG."""

    def __init__(
        self,
        mesh: Q4Mesh | Hex8Mesh,
        fixed_dofs: Sequence[int],
        material: dict[str, float],
        device: DeviceLike = None,
        dtype: torch.dtype = torch.float64,
        use_fused: bool = False,
        preconditioner: str = "jacobi",
        springs: "Sequence[float] | None" = None,
        mgcg_coarsest_dtype: str | torch.dtype = "float64",
        mgcg_semi_coarsen: bool = False,
    ) -> None:
        super().__init__()
        if dtype != torch.float64:
            raise ValueError("MechanicsOperator requires torch.float64")
        if preconditioner not in ("jacobi", "mgcg", "mgcg_masked", "amgcg"):
            raise ValueError(f"Unknown elasticity preconditioner {preconditioner!r}")
        from hgto.fem.solvers.mgcg import resolve_coarsest_dtype

        self.mgcg_coarsest_dtype = resolve_coarsest_dtype(mgcg_coarsest_dtype)
        self.mgcg_semi_coarsen = bool(mgcg_semi_coarsen)
        self.mgcg_stamp: dict[str, object] | None = None
        self.mgcg_fp32_fallbacks = 0
        if mesh.coords.ndim != 2 or int(mesh.coords.shape[1]) not in (2, 3):
            raise ValueError("mesh coordinates must have shape (Nn, 2) or (Nn, 3)")
        n_dim = int(mesh.coords.shape[1])
        if preconditioner == "amgcg" and n_dim != 2:
            raise ValueError("SA-AMG currently supports Q4 plane stress only")
        if preconditioner == "mgcg" and not mesh.is_regular:
            raise ValueError(
                "MGCG requires the binding structured grid "
                "(Q4Mesh.is_regular); compact/imported meshes use jacobi"
            )
        if preconditioner == "mgcg_masked":
            from hgto.fem.solvers.masked_mg import CartesianEmbedding

            self.cartesian_embedding = CartesianEmbedding.from_mesh(mesh)
        if preconditioner in ("mgcg", "mgcg_masked", "amgcg") and springs is not None:
            raise ValueError(
                "MGCG does not yet carry nodal grounded springs (mechanism "
                "cases); use the jacobi preconditioner"
            )
        required = ("E0", "Emin", "nu", "p")
        missing = [key for key in required if key not in material]
        if missing:
            raise KeyError("material is missing keys: " + ", ".join(missing))

        E0 = float(material["E0"])
        Emin = float(material["Emin"])
        nu = float(material["nu"])
        p = float(material["p"])
        if not (E0 > 0.0 and 0.0 < Emin <= E0):
            raise ValueError("material requires E0 > 0 and 0 < Emin <= E0")
        if not (-1.0 < nu < 0.5):
            raise ValueError(
                "plane-stress Poisson ratio must be in (-1, 0.5)"
                if n_dim == 2
                else "3D Poisson ratio must be in (-1, 0.5)"
            )
        if p <= 0.0:
            raise ValueError("SIMP exponent p must be positive")
        if mesh.thickness <= 0.0:
            raise ValueError("mesh thickness must be positive")

        # Construction-time dimension constants. Both
        # binding quadrature rules collocate one Gauss point per local node.
        nodes_per_element = 4 if n_dim == 2 else 8
        n_gauss = nodes_per_element
        n_voigt = 3 if n_dim == 2 else 6

        target_device = torch.device("cpu") if device is None else torch.device(device)
        coords = torch.as_tensor(mesh.coords, dtype=dtype, device=target_device)
        econn = torch.as_tensor(mesh.econn, dtype=torch.long, device=target_device)
        if coords.shape != (mesh.n_nodes, n_dim):
            raise ValueError(f"mesh coordinates must have shape (Nn, {n_dim})")
        if econn.shape != (mesh.n_elements, nodes_per_element):
            raise ValueError(f"mesh connectivity must have shape (Ne, {nodes_per_element})")

        fixed = torch.as_tensor(fixed_dofs, dtype=torch.long, device=target_device).reshape(-1)
        if fixed.numel() > 0:
            if bool(torch.any((fixed < 0) | (fixed >= mesh.n_dof)).item()):
                raise ValueError("fixed_dofs contains an out-of-range DOF")
            fixed = torch.unique(fixed, sorted=True)
        free_mask = torch.ones(mesh.n_dof, dtype=torch.bool, device=target_device)
        free_mask[fixed] = False
        if not bool(torch.any(free_mask).item()):
            raise ValueError("at least one free DOF is required")

        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        self.n_dof = mesh.n_dof
        self.n_dim = n_dim
        self.nodes_per_element = nodes_per_element
        self.n_gauss = n_gauss
        self.n_voigt = n_voigt
        self.use_fused = bool(use_fused)
        self.preconditioner = preconditioner
        # Structured-grid metadata for the MGCG hierarchy (plain ints; the
        # jacobi path never reads these).
        self.mesh_nelx = int(mesh.nelx)
        self.mesh_nely = int(mesh.nely)
        self.mesh_nelz = int(getattr(mesh, "nelz", 0))
        self.mesh_is_regular = bool(mesh.is_regular)

        self.register_buffer("coords", coords)
        self.register_buffer("econn", econn)
        self.register_buffer("fixed_dofs", fixed)
        self.register_buffer("free_dof_mask", free_mask)
        # Long-index twin of the free-DOF mask: gathers / scatters through it
        # are plain index_select / index_put kernels with no hidden
        # ``nonzero()`` host sync (boolean-mask indexing reads the count
        # back every call — 4 syncs per reduced matvec on cuda).  Same
        # values, bitwise, on every device.
        self.register_buffer(
            "free_dof_index",
            torch.nonzero(free_mask, as_tuple=False).reshape(-1),
        )
        self.register_buffer("E0", torch.tensor(E0, dtype=dtype, device=target_device))
        self.register_buffer("Emin", torch.tensor(Emin, dtype=dtype, device=target_device))
        self.register_buffer("nu", torch.tensor(nu, dtype=dtype, device=target_device))
        self.register_buffer("p", torch.tensor(p, dtype=dtype, device=target_device))
        self.register_buffer(
            "thickness", torch.tensor(mesh.thickness, dtype=dtype, device=target_device)
        )

        if springs is not None:
            spring_vector = torch.as_tensor(springs, dtype=dtype, device=target_device).reshape(-1)
            if spring_vector.shape != (mesh.n_dof,):
                raise ValueError("springs must have one stiffness per DOF")
            if bool(torch.any(spring_vector < 0.0).item()) or not bool(
                torch.all(torch.isfinite(spring_vector)).item()
            ):
                raise ValueError("spring stiffnesses must be finite and >= 0")
            self.register_buffer("spring_stiffness", spring_vector)
        else:
            self.spring_stiffness = None

        tables = geometry_tables(coords, econn)
        self.register_buffer("dN_dx", tables["dN_dx"])
        self.register_buffer("detJ", tables["detJ"])
        self.register_buffer("wdetJ", tables["wdetJ"])
        self.register_buffer("integration_weights", tables["wdetJ"] * self.thickness)
        self.register_buffer("Ke0", self._preintegrate_unit_stiffness())

        # Explicit construction-time fp64 conversion is binding under .
        self.double()

    @property
    def device(self) -> torch.device:
        return self.coords.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coords.dtype

    def _preintegrate_unit_stiffness(self) -> torch.Tensor:
        """Unit-modulus per-element stiffness ``K_e^0`` (fused hot path + diag)."""
        if self.n_dim == 2:
            B = self.dN_dx.new_zeros((self.n_elements, 4, 3, 8))
            B[:, :, 0, 0::2] = self.dN_dx[..., 0]
            B[:, :, 1, 1::2] = self.dN_dx[..., 1]
            B[:, :, 2, 0::2] = self.dN_dx[..., 1]
            B[:, :, 2, 1::2] = self.dN_dx[..., 0]
            C0 = plane_stress_matrix(1.0, self.nu, self.dN_dx)
            return torch.einsum("egki,kl,eglj,eg->eij", B, C0, B, self.integration_weights)
        # Hex8 branch: B (Ne, 8, 6, 24), Voigt [xx, yy, zz, yz, xz, xy] with
        # engineering shear, interleaved dofs (ux, uy, uz) via 3-stride slices.
        B = self.dN_dx.new_zeros((self.n_elements, 8, 6, 24))
        B[:, :, 0, 0::3] = self.dN_dx[..., 0]
        B[:, :, 1, 1::3] = self.dN_dx[..., 1]
        B[:, :, 2, 2::3] = self.dN_dx[..., 2]
        B[:, :, 3, 1::3] = self.dN_dx[..., 2]
        B[:, :, 3, 2::3] = self.dN_dx[..., 1]
        B[:, :, 4, 0::3] = self.dN_dx[..., 2]
        B[:, :, 4, 2::3] = self.dN_dx[..., 0]
        B[:, :, 5, 0::3] = self.dN_dx[..., 1]
        B[:, :, 5, 1::3] = self.dN_dx[..., 0]
        C0 = isotropic_C_3d(1.0, self.nu, self.dN_dx)
        return torch.einsum("egki,kl,eglj,eg->eij", B, C0, B, self.integration_weights)

    # ---- boundary checks ----

    def _check_rho(self, rho: torch.Tensor) -> None:
        if rho.shape != (self.n_elements,):
            raise ValueError("rho must have shape (Ne,)")
        if rho.dtype != self.dtype or rho.device != self.device:
            raise TypeError("rho must match the operator dtype and device")

    def _check_vector_field(self, value: torch.Tensor, name: str) -> None:
        if value.ndim != 3 or value.shape[1:] != (self.n_nodes, self.n_dim):
            raise ValueError(f"{name} must have shape (L, Nn, {self.n_dim})")
        if value.dtype != self.dtype or value.device != self.device:
            raise TypeError(f"{name} must match the operator dtype and device")

    def _free_field_mask(self) -> torch.Tensor:
        return self.free_dof_mask.reshape(1, self.n_nodes, self.n_dim)

    def _masked(self, value: torch.Tensor) -> torch.Tensor:
        return value * self._free_field_mask()

    # ---- material interpolation ----

    def element_youngs(self, rho: torch.Tensor) -> torch.Tensor:
        """SIMP interpolation including the void stiffness floor."""
        self._check_rho(rho)
        return self.Emin + torch.pow(rho, self.p) * (self.E0 - self.Emin)

    # ---- matvec paths ----

    def _chain_matvec(self, value: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        strain = gauss_strains(value, self.econn, self.dN_dx)
        stress = gauss_stress(strain, self.element_youngs(rho), self.nu)
        return internal_force(
            stress, self.econn, self.dN_dx, self.integration_weights, n_nodes=self.n_nodes
        )

    def _fused_matvec(self, value: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        element_u = value[:, self.econn, :].reshape(
            value.shape[0], self.n_elements, self.nodes_per_element * self.n_dim
        )
        element_force = torch.einsum("eij,lej->lei", self.Ke0, element_u)
        element_force = element_force * self.element_youngs(rho)[None, :, None]
        force = value.new_zeros((value.shape[0], self.n_nodes, self.n_dim))
        force.index_add_(
            1, self.econn.reshape(-1), element_force.reshape(value.shape[0], -1, self.n_dim)
        )
        return force

    def _raw_matvec(self, value: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if self.use_fused:
            return self._fused_matvec(value, rho)
        return self._chain_matvec(value, rho)

    # ---- operator API ----

    def _spring_force(self, u_masked: torch.Tensor) -> torch.Tensor:
        """Grounded-spring nodal force ``S u`` (zero when no springs)."""
        return u_masked * self.spring_stiffness.reshape(1, self.n_nodes, self.n_dim)

    def residual(self, u: torch.Tensor, rho: torch.Tensor, f_ext: torch.Tensor) -> torch.Tensor:
        """Reduced-space ``f_int - f_ext``, zero at prescribed DOFs."""
        self._check_vector_field(u, "u")
        self._check_vector_field(f_ext, "f_ext")
        if u.shape[0] != f_ext.shape[0]:
            raise ValueError("u and f_ext load counts differ")
        self._check_rho(rho)
        u_masked = self._masked(u)
        internal = self._raw_matvec(u_masked, rho)
        if self.spring_stiffness is not None:
            internal = internal + self._spring_force(u_masked)
        return self._masked(internal - f_ext)

    def energy(
        self, u: torch.Tensor, rho: torch.Tensor, f_ext: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Explicitly named scalar energies: no ambiguous 'objective'."""
        self._check_vector_field(u, "u")
        self._check_vector_field(f_ext, "f_ext")
        if u.shape[0] != f_ext.shape[0]:
            raise ValueError("u and f_ext load counts differ")
        self._check_rho(rho)
        u_free = self._masked(u)
        strain = gauss_strains(u_free, self.econn, self.dN_dx)
        energy_gp = strain_energy_gp(strain, self.element_youngs(rho), self.nu)
        elastic = torch.sum(energy_gp * self.integration_weights[None, :, :])
        if self.spring_stiffness is not None:
            elastic = elastic + 0.5 * torch.sum(self._spring_force(u_free) * u_free)
        external_work = torch.sum(u_free * f_ext)
        return {
            "strain_energy": elastic,
            "potential": elastic - external_work,
            "compliance": external_work,
        }

    def jvp_u(self, u_dir: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Apply the reduced tangent ``K(rho) (+ S)`` matrix-free."""
        self._check_vector_field(u_dir, "u_dir")
        self._check_rho(rho)
        u_masked = self._masked(u_dir)
        product = self._raw_matvec(u_masked, rho)
        if self.spring_stiffness is not None:
            product = product + self._spring_force(u_masked)
        return self._masked(product)

    def vjp_u(self, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Transpose tangent; the linear-elastic K is symmetric."""
        return self.jvp_u(v, rho)

    def vjp_rho(self, u: torch.Tensor, rho: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """``lam^T dR/drho`` per element (adjoint building block, )."""
        self._check_vector_field(u, "u")
        self._check_vector_field(lam, "lam")
        if u.shape[0] != lam.shape[0]:
            raise ValueError("u and lam load counts differ")
        self._check_rho(rho)
        strain_u = gauss_strains(self._masked(u), self.econn, self.dN_dx)
        strain_lam = gauss_strains(self._masked(lam), self.econn, self.dN_dx)
        stress_u_unit = gauss_stress(strain_u, torch.ones_like(rho), self.nu)
        cross_gp = torch.sum(strain_lam * stress_u_unit, dim=-1)
        cross_element = torch.sum(cross_gp * self.integration_weights[None, :, :], dim=(0, 2))
        dE_drho = self.p * torch.pow(rho, self.p - 1.0) * (self.E0 - self.Emin)
        return dE_drho * cross_element

    def diag_stiffness(self, rho: torch.Tensor) -> torch.Tensor:
        """Reduced matrix diagonal for Jacobi preconditioning."""
        self._check_rho(rho)
        element_diag = torch.diagonal(self.Ke0, dim1=1, dim2=2)
        element_diag = element_diag * self.element_youngs(rho)[:, None]
        local_dofs = (
            self.n_dim * self.econn[:, :, None]
            + torch.arange(self.n_dim, dtype=torch.long, device=self.device)[None, None, :]
        )
        diagonal = rho.new_zeros(self.n_dof)
        diagonal.index_add_(0, local_dofs.reshape(-1), element_diag.reshape(-1))
        if self.spring_stiffness is not None:
            diagonal = diagonal + self.spring_stiffness
        return diagonal * self.free_dof_mask

    # ---- solves ----

    def _reduced_matvec_fn(self, rho: torch.Tensor):
        free_index = self.free_dof_index
        n_dof, n_nodes, n_dim = self.n_dof, self.n_nodes, self.n_dim

        def matvec(x: torch.Tensor) -> torch.Tensor:
            full = x.new_zeros(n_dof)
            full[free_index] = x
            product = self.jvp_u(full.reshape(1, n_nodes, n_dim), rho)
            return product.reshape(-1)[free_index]

        return matvec

    @torch.no_grad()
    def _solve_displacements(
        self,
        rho: torch.Tensor,
        rhs: torch.Tensor,
        u0: torch.Tensor | None,
        rtol: float,
        max_iter: int,
        allow_direct_fallback: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_rho(rho)
        self._check_vector_field(rhs, "rhs")
        if u0 is None:
            initial = torch.zeros_like(rhs)
        else:
            self._check_vector_field(u0, "u0")
            if u0.shape != rhs.shape:
                raise ValueError("u0 and rhs must have identical shapes")
            initial = self._masked(u0)

        matvec = self._reduced_matvec_fn(rho)
        free_index = self.free_dof_index
        sparse_K = None
        mg_precond = None
        diagonal = None
        if self.preconditioner in ("mgcg", "mgcg_masked", "amgcg"):
            from hgto.fem.solvers.mgcg import MGHierarchy, preconditioned_cg

            # Optional hierarchy reuse keeps the preconditioner fixed within
            # each CG solve. The operator and residual use the current density.
            reuse_steps = int(getattr(self, "preconditioner_reuse_steps", 1))
            youngs = self.element_youngs(rho)
            cache = getattr(self, "_mg_hierarchy_cache", None)
            use_cached = reuse_steps > 1 and cache is not None and cache[2] < reuse_steps
            if use_cached:
                old = cache[1]
                ratio = torch.maximum(youngs / old, old / youngs).max()
                use_cached = float(ratio) <= float(getattr(self, "preconditioner_reuse_ratio", 4.0))
            if use_cached:
                hierarchy = cache[0]
                self._mg_hierarchy_cache = (hierarchy, cache[1], cache[2] + 1)
            else:
                if self.preconditioner == "amgcg":
                    from hgto.fem.solvers.algebraic_mg import AlgebraicMGHierarchy

                    hierarchy = AlgebraicMGHierarchy(self, youngs)
                elif self.preconditioner == "mgcg_masked":
                    from hgto.fem.solvers.masked_mg import MaskedMGHierarchy

                    hierarchy = MaskedMGHierarchy(self, youngs)
                else:
                    hierarchy = MGHierarchy(
                        self.Ke0,
                        youngs,
                        self.econn,
                        self.free_dof_mask,
                        self.mesh_nelx,
                        self.mesh_nely,
                        nelz=self.mesh_nelz if self.n_dim == 3 else None,
                        coarsest_dtype=self.mgcg_coarsest_dtype,
                        semi_coarsen=self.mgcg_semi_coarsen,
                    )
                self.mgcg_fp32_fallbacks += int(hierarchy.coarsest_fp32_fallback)
                if reuse_steps > 1:
                    self._mg_hierarchy_cache = (hierarchy, youngs.clone(), 1)
            self.mg_hierarchy_builds = int(getattr(self, "mg_hierarchy_builds", 0)) + int(
                not use_cached
            )
            self.mg_hierarchy_reuses = int(getattr(self, "mg_hierarchy_reuses", 0)) + int(
                use_cached
            )
            stamp = hierarchy.stamp()
            stamp["mgcg_fp32_fallbacks"] = int(self.mgcg_fp32_fallbacks)
            self.mgcg_stamp = stamp
            n_dof = self.n_dof

            def mg_precond(residual_reduced: torch.Tensor) -> torch.Tensor:
                full = residual_reduced.new_zeros(n_dof)
                full[free_index] = residual_reduced
                return hierarchy.apply(full)[free_index]

            if getattr(self, "cuda_graph_preconditioner", False):
                from hgto.fem.solvers.cuda_action import CapturedAction

                captured = getattr(hierarchy, "_captured_reduced_action", None)
                if captured is None:
                    example = torch.zeros(free_index.numel(), device=self.device, dtype=self.dtype)
                    captured = CapturedAction(mg_precond, example)
                    hierarchy._captured_reduced_action = captured
                mg_precond = captured
        else:
            # The Jacobi diagonal is dead on the mgcg path (only
            # ``pcg_callback`` reads it): build it only where it is used.
            diagonal = self.diag_stiffness(rho)[free_index]

        solutions, residuals, iteration_counts, converged_flags, fallback_flags = [], [], [], [], []
        for load_index in range(rhs.shape[0]):
            rhs_reduced = rhs[load_index].reshape(-1)[free_index]
            x0_reduced = initial[load_index].reshape(-1)[free_index].clone()
            if mg_precond is not None:
                x, residual_rel, iterations, converged = preconditioned_cg(
                    matvec, rhs_reduced, mg_precond, x0_reduced, rtol, max_iter
                )
            else:
                x, residual_rel, iterations, converged = pcg_callback(
                    matvec, rhs_reduced, diagonal, x0_reduced, rtol, max_iter
                )
            fallback_used = False
            if not converged and allow_direct_fallback:
                # Certified sparse direct rescue for high-contrast stalls.
                if sparse_K is None:
                    sparse_K = assemble_sparse_stiffness(
                        self.Ke0, self.element_youngs(rho), self.econn, self.n_dof
                    )
                    if self.spring_stiffness is not None:
                        import scipy.sparse as _sparse

                        sparse_K = sparse_K + _sparse.diags(self.spring_stiffness.cpu().numpy())
                full_rhs = rhs[load_index].reshape(-1)
                x_full, residual_rel, converged = direct_solve_reduced(
                    sparse_K, full_rhs, self.free_dof_mask, rtol
                )
                x = x_full[free_index]
                fallback_used = True
            if not converged:
                raise SolveFailure(
                    f"LE state solve failed: residual_rel={float(residual_rel):.6e} "
                    f"after {iterations} iterations (load {load_index})",
                    kind="direct"
                    if fallback_used
                    else ("mgcg" if mg_precond is not None else "pcg"),
                    residual_rel=float(residual_rel),
                    iterations=iterations,
                )
            full = x.new_zeros(self.n_dof)
            full[free_index] = x
            solutions.append(full.reshape(self.n_nodes, self.n_dim))
            residuals.append(residual_rel if residual_rel.ndim == 0 else residual_rel.reshape(()))
            iteration_counts.append(iterations)
            converged_flags.append(converged)
            fallback_flags.append(fallback_used)
        return (
            torch.stack(solutions, dim=0),
            torch.stack(residuals, dim=0),
            torch.tensor(iteration_counts, dtype=torch.long, device=self.device),
            torch.tensor(converged_flags, dtype=torch.bool, device=self.device),
            torch.tensor(fallback_flags, dtype=torch.bool, device=self.device),
        )

    @torch.no_grad()
    def evaluate_linear_state(
        self,
        rho: torch.Tensor,
        f_ext: torch.Tensor,
        u: torch.Tensor,
        residual_rel: torch.Tensor,
        iterations: torch.Tensor,
        converged: torch.Tensor,
        fallback_used: torch.Tensor | None = None,
    ) -> MechanicsState:
        """Evaluate linear state fields at ``u`` with externally supplied certificates."""
        self._check_rho(rho)
        self._check_vector_field(f_ext, "f_ext")
        self._check_vector_field(u, "u")
        if u.shape != f_ext.shape:
            raise ValueError("u and f_ext must have identical shapes")
        expected = (f_ext.shape[0],)
        if (
            residual_rel.shape != expected
            or iterations.shape != expected
            or converged.shape != expected
        ):
            raise ValueError("certificates must have shape (L,)")
        if residual_rel.dtype != self.dtype or residual_rel.device != self.device:
            raise TypeError("residual certificates must match operator dtype and device")
        if iterations.dtype != torch.long or converged.dtype != torch.bool:
            raise TypeError("iterations must be torch.long and converged torch.bool")
        strain = gauss_strains(u, self.econn, self.dN_dx)
        E_e = self.element_youngs(rho)
        stress = gauss_stress(strain, E_e, self.nu)
        unit_gp = unit_strain_energy_gp(strain, self.nu)
        physical_gp = 0.5 * unit_gp * E_e[None, :, None]
        f_int = self._raw_matvec(u, rho)
        if self.spring_stiffness is not None:
            f_int = f_int + self._spring_force(self._masked(u))
        compliance = torch.sum(u * f_ext)
        return MechanicsState(
            u=u,
            strain_gp=strain,
            stress_gp=stress,
            unit_strain_energy_gp=unit_gp,
            strain_energy_gp=physical_gp,
            f_int=f_int,
            residual_rel=residual_rel,
            iterations=iterations,
            converged=converged,
            compliance=compliance,
            fallback_used=fallback_used,
        )

    @torch.no_grad()
    def solve_state(
        self,
        rho: torch.Tensor,
        f_ext: torch.Tensor,
        u0: torch.Tensor | None = None,
        rtol: float = 1.0e-8,
        max_iter: int = 20000,
        allow_direct_fallback: bool = False,
    ) -> MechanicsState:
        """Solve all load cases with fp64 Jacobi-PCG; certified, typed failures."""
        u, residual_rel, iterations, converged, fallback = self._solve_displacements(
            rho, f_ext, u0, rtol, max_iter, allow_direct_fallback
        )
        return self.evaluate_linear_state(
            rho, f_ext, u, residual_rel, iterations, converged, fallback_used=fallback
        )

    @torch.no_grad()
    def solve_adjoint(
        self,
        rho: torch.Tensor,
        rhs: torch.Tensor,
        u0: torch.Tensor | None = None,
        rtol: float = 1.0e-8,
        max_iter: int = 20000,
        allow_direct_fallback: bool = False,
    ) -> MechanicsState:
        """Adjoint solve for an explicit tensor RHS (symmetric LE: same system).

        callers pass ``rhs = -dJ/du`` as a tensor, never an
        objective-name string.
        """
        return self.solve_state(
            rho,
            rhs,
            u0=u0,
            rtol=rtol,
            max_iter=max_iter,
            allow_direct_fallback=allow_direct_fallback,
        )

    @torch.no_grad()
    def compliance_sensitivity(self, rho: torch.Tensor, state: MechanicsState) -> torch.Tensor:
        """Self-adjoint SIMP compliance gradient.

        consumes the UNHALVED ``unit_strain_energy_gp`` — using the
        physical 1/2-form here would halve every sensitivity.
        """
        self._check_rho(rho)
        unit_gp = state.unit_strain_energy_gp
        if unit_gp.ndim != 3 or unit_gp.shape[1:] != (self.n_elements, self.n_gauss):
            raise ValueError(f"state.unit_strain_energy_gp must have shape (L, Ne, {self.n_gauss})")
        if unit_gp.dtype != self.dtype or unit_gp.device != self.device:
            raise TypeError("state tensors must match the operator dtype and device")
        integrated_unit = torch.sum(unit_gp * self.integration_weights[None, :, :], dim=(0, 2))
        return -self.p * torch.pow(rho, self.p - 1.0) * (self.E0 - self.Emin) * integrated_unit
