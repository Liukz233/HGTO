"""Load a conforming mesh and its support/load metadata."""

from pathlib import Path
import json
import numpy as np
from hgto.fem.mesh.q4 import Q4Mesh
from hgto.fem.mesh.tri3 import Tri3Mesh


def load_case(folder):
    """Return metadata, arrays and a Q4 or Tri3 mesh chosen by the connectivity width."""
    folder = Path(folder)
    meta = json.loads((folder / "case.json").read_text())
    arrays = dict(np.load(folder / "geometry.npz"))
    mesh_type = {3: Tri3Mesh, 4: Q4Mesh}.get(arrays["econn"].shape[1])
    if mesh_type is None:
        raise ValueError("Irregular domains use Tri3 or Q4 connectivity")
    mesh = mesh_type(nelx=0, nely=0, coords=arrays["coords"], econn=arrays["econn"])
    return meta, arrays, mesh
