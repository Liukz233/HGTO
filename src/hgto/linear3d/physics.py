"""Three-dimensional graph equilibrium and compliance sensitivities."""

import numpy as np
import torch
from hgto.fem import MechanicsOperator
from hgto.topopt.physics.linear import LinearCompliance


class Physics:
    def __init__(self, problem, mesh, device):
        self.operator = MechanicsOperator(
            mesh,
            problem.fixed_dofs,
            dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
            device=device,
            use_fused=True,
            preconditioner="mgcg",
            mgcg_coarsest_dtype="float32",
            mgcg_semi_coarsen=True,
        )
        forces = torch.tensor(problem.force, device=device).reshape(1, -1, 3)
        self.physics = LinearCompliance(
            self.operator,
            forces,
            rtol=1e-8,
            max_iter=20000,
            allow_direct_fallback=False,
            bw_rtol=1e-8,
            volumes=np.ones(problem.n_elem),
            volume_fraction=problem.volfrac,
        )
        self.calls = 0
        self.max_residual = 0.0

    def evaluate(self, rho):
        a = self.physics.analyze(rho.detach(), self.calls)
        self.calls += 1
        self.max_residual = max(self.max_residual, float(a.diagnostics["residual_rel"]))
        if a.diagnostics["state_backend"] != "hypergraph":
            raise RuntimeError("Unexpected state backend")
        return float(a.objective), a.dJ_drho.detach()
