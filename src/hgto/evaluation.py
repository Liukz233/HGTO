"""Independent CPU checks on saved physical densities."""

import json
from pathlib import Path
import numpy as np


def evaluate_design(directory):
    directory = Path(directory)
    if (directory / "protocol.json").exists():
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["physics"] == "nh":
            from hgto.nonlinear.evaluation import evaluate_nh

            return evaluate_nh(directory)
        raise ValueError(
            "Use compare-plastic to evaluate both designs through a common loading history"
        )
    from hgto.reference import ScikitFEMElasticity

    with np.load(directory / "geometry.npz", allow_pickle=False) as data:
        values = {key: data[key] for key in ("coords", "cells", "fixed", "forces")}
    rho = np.load(directory / "rho.npy", allow_pickle=False)
    solver = ScikitFEMElasticity(**values, solver="scipy")
    try:
        compliance, _, _ = solver.evaluate(rho)
        v = solver.element_volumes
        return dict(
            compliance=float(compliance),
            volume=float(v @ rho / v.sum()),
            relative_residual=float(solver.last_residual),
        )
    finally:
        solver.close()
