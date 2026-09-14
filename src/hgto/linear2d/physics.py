"""Public mechanics interface: graph element operator and warm MG-PCG states."""

import numpy as np
import torch


def make_mesh(problem):
    from hgto.fem.mesh.q4 import Q4Mesh

    ny, nx = problem.shape
    return Q4Mesh(nx, ny, problem.coords, problem.cells, thickness=1.0)


class GraphElasticity:
    def __init__(self, problem, mesh, device="cuda"):
        from hgto.fem import MechanicsOperator
        from hgto.topopt.physics.linear import LinearCompliance

        mode = "mgcg" if mesh.is_regular else "amgcg"
        if not mesh.is_regular:
            from hgto.fem.solvers.masked_mg import CartesianEmbedding

            try:
                CartesianEmbedding.from_mesh(mesh)
                mode = "mgcg_masked"
            except ValueError:
                pass
        self.operator = MechanicsOperator(
            mesh,
            problem.fixed,
            dict(E0=1.0, Emin=1e-6, nu=0.3, p=3.0),
            device=device,
            use_fused=True,
            preconditioner=mode,
            mgcg_coarsest_dtype="float32",
            mgcg_semi_coarsen=True,
        )
        # Capture only the fixed preconditioner; current stiffness and the
        # true CG residual remain freshly evaluated at every design update.
        if str(device).startswith("cuda"):
            self.operator.preconditioner_reuse_steps = 10
            self.operator.preconditioner_reuse_ratio = 2.0
            self.operator.cuda_graph_preconditioner = True
        forces = torch.as_tensor(
            problem.forces.T.copy(), device=device, dtype=torch.float64
        ).reshape(-1, len(problem.coords), 2)
        self.physics = LinearCompliance(
            self.operator,
            forces,
            rtol=1e-8,
            max_iter=20000,
            allow_direct_fallback=False,
            bw_rtol=1e-8,
            volumes=mesh.element_volumes(),
            volume_fraction=problem.volume_fraction,
        )
        self.calls = 0
        self.max_residual = 0.0
        self.iterations = []

    def evaluate(self, rho):
        analysis = self.physics.analyze(rho.detach(), self.calls)
        self.calls += 1
        d = analysis.diagnostics
        self.max_residual = max(self.max_residual, float(d["residual_rel"]))
        self.iterations.append(int(d["solver_iterations"]))
        assert d["state_backend"] == "hypergraph"
        return float(analysis.objective), analysis.dJ_drho.detach()
