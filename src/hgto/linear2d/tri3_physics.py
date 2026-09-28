"""Linear Tri3 elasticity on the GPU: hyperedge operations and a direct cuDSS solve.

Element matrices are cached for unit stiffness and scaled by the SIMP
modulus at every design update. The assembled free-DOF stiffness is
factorized by cuDSS; the compliance sensitivity is exact for this model."""

import time

import numpy as np
import torch

from hgto.fem.mesh.tri3 import triangle_unit_stiffness
from hgto.fem.solvers.cudss_tangent import solve_cuda_tangent


class Tri3Elasticity:
    def __init__(self, mesh, fixed, force, device="cuda:0", nu=0.3, Emin=1e-6):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.n_elements = mesh.n_elements
        self.n_dof = mesh.n_dof
        self.econn = torch.as_tensor(mesh.econn, device=self.device)
        self.Ke0 = torch.as_tensor(triangle_unit_stiffness(mesh, nu), device=self.device)
        self.edofs = (2 * self.econn[:, :, None] + torch.arange(2, device=self.device)).reshape(
            -1, 6
        )
        self.free_dof_mask = torch.ones(self.n_dof, device=self.device, dtype=torch.bool)
        self.free_dof_mask[torch.as_tensor(fixed, device=self.device)] = False
        self.force = torch.as_tensor(np.asarray(force).reshape(-1), device=self.device)
        self.penalty = 3.0
        self.Emin = Emin
        self.max_residual = self.last_residual = 0.0
        self.calls = 0
        self.state_seconds = self.sensitivity_seconds = 0.0

    def set_penalty(self, p):
        self.penalty = float(p)

    def _synchronized_time(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def evaluate(self, rho):
        started = self._synchronized_time()
        rho = rho.detach().to(self.device)
        modulus = self.Emin + (1 - self.Emin) * rho**self.penalty
        Ke = self.Ke0 * modulus[:, None, None]

        def matvec(v):
            full = v.new_zeros(self.n_dof)
            full[self.free_dof_mask] = v
            element = torch.einsum("eij,ej->ei", Ke, full[self.edofs])
            out = v.new_zeros(self.n_dof)
            out.index_add_(0, self.edofs.reshape(-1), element.reshape(-1))
            return out[self.free_dof_mask]

        reduced, residual, _, ok = solve_cuda_tangent(
            self, Ke, self.force[self.free_dof_mask], matvec, 1e-10
        )
        if not ok:
            raise RuntimeError(f"Tri3 state residual {residual}")
        u = rho.new_zeros(self.n_dof)
        u[self.free_dof_mask] = reduced
        C = float(u @ self.force)
        now = self._synchronized_time()
        self.state_seconds += now - started
        ue = u[self.edofs]
        energy = torch.einsum("ei,eij,ej->e", ue, self.Ke0, ue)
        g = -self.penalty * (1 - self.Emin) * rho ** (self.penalty - 1) * energy
        self.sensitivity_seconds += self._synchronized_time() - now
        self.last_u = u
        self.last_residual = residual
        self.max_residual = max(residual, self.max_residual)
        self.calls += 1
        return C, g

    def close(self):
        if hasattr(self, "_cudss_tangent"):
            self._cudss_tangent.close()
