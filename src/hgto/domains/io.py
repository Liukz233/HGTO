"""Load a conforming mesh and its support/load metadata."""

from pathlib import Path
import json
import numpy as np
from hgto.fem.mesh.q4 import Q4Mesh


def load_case(folder):
    folder = Path(folder)
    meta = json.loads((folder / "case.json").read_text())
    arrays = dict(np.load(folder / "geometry.npz"))
    mesh = Q4Mesh(nelx=0, nely=0, coords=arrays["coords"], econn=arrays["econn"])
    return meta, arrays, mesh
