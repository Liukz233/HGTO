"""Vectorized assembled Q4 plane-stress mechanics with true free-DOF residuals."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def q4_stiffness(xy, nu=0.3):
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    D = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]]) / (1 - nu**2)
    K = np.zeros((8, 8))
    for xi, eta in corners / np.sqrt(3.0):
        grad = (
            np.c_[
                corners[:, 0] * (1 + eta * corners[:, 1]), corners[:, 1] * (1 + xi * corners[:, 0])
            ]
            / 4
        )
        J = xy.T @ grad
        physical = grad @ np.linalg.inv(J)
        B = np.zeros((3, 8))
        B[0, ::2] = physical[:, 0]
        B[1, 1::2] = physical[:, 1]
        B[2, ::2] = physical[:, 1]
        B[2, 1::2] = physical[:, 0]
        K += B.T @ D @ B * np.linalg.det(J)
    return (K + K.T) / 2


class Elasticity:
    def __init__(self, problem, emin=1e-6, penalty=3.0):
        self.problem = problem
        self.emin = emin
        self.penalty = penalty
        self.ke = q4_stiffness(problem.coords[problem.cells[0]])
        self.dofs = np.empty((problem.n_elements, 8), dtype=int)
        self.dofs[:, ::2] = 2 * problem.cells
        self.dofs[:, 1::2] = 2 * problem.cells + 1
        self.rows = np.repeat(self.dofs, 8, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 8)).ravel()
        self.free = np.setdiff1d(np.arange(problem.n_dofs), problem.fixed)
        self.last_residual = 0.0
        self.solves = 0

    def evaluate(self, rho):
        E = self.emin + (1 - self.emin) * rho**self.penalty
        K = sparse.coo_matrix(
            ((E[:, None] * self.ke.ravel()).ravel(), (self.rows, self.cols)),
            shape=(self.problem.n_dofs, self.problem.n_dofs),
        ).tocsc()
        reduced = K[self.free][:, self.free]
        u = np.zeros_like(self.problem.forces)
        u[self.free] = splu(reduced).solve(self.problem.forces[self.free])
        resid = reduced @ u[self.free] - self.problem.forces[self.free]
        self.last_residual = float(
            np.linalg.norm(resid) / np.linalg.norm(self.problem.forces[self.free])
        )
        assert self.last_residual < 1e-7, self.last_residual
        ue = u[self.dofs]
        energy = np.einsum("eil,ij,ejl->e", ue, self.ke, ue)
        C = float(np.sum(self.problem.forces * u))
        gradient = -self.penalty * (1 - self.emin) * rho ** (self.penalty - 1) * energy
        self.solves += 1
        return C, gradient, u
