"""Q4 and Hex8 meshes, reference quadrature, graph adjacency and mesh import."""

from hgto.fem.mesh.dual_graph import build_element_dual_graph
from hgto.fem.mesh.generators import (
    CoordinateMap,
    annulus_sector,
    graded_q4,
    lbracket_q4,
    mapped_q4,
    staircase_slit_q4,
)
from hgto.fem.mesh.hex8 import Hex8Mesh, distorted_hex8, dof_ids_3d, structured_hex8
from hgto.fem.mesh.io import read_abaqus_inp_quads, read_gmsh_quads
from hgto.fem.mesh.q4 import Q4Mesh, distorted_q4, dof_ids, structured_q4
from hgto.fem.mesh.quadrature import (
    GAUSS_G,
    GAUSS_POINTS,
    GAUSS_POINTS_3D,
    GAUSS_WEIGHTS,
    GAUSS_WEIGHTS_3D,
    LOCAL_NODES_XI,
    LOCAL_NODES_XI_3D,
    hex8_reference_tables,
    q4_reference_tables,
    shape_functions,
    shape_functions_3d,
    shape_gradients,
    shape_gradients_3d,
)

__all__ = [
    "CoordinateMap",
    "GAUSS_G",
    "GAUSS_POINTS",
    "GAUSS_POINTS_3D",
    "GAUSS_WEIGHTS",
    "GAUSS_WEIGHTS_3D",
    "Hex8Mesh",
    "LOCAL_NODES_XI",
    "LOCAL_NODES_XI_3D",
    "Q4Mesh",
    "annulus_sector",
    "build_element_dual_graph",
    "distorted_hex8",
    "distorted_q4",
    "dof_ids",
    "dof_ids_3d",
    "graded_q4",
    "hex8_reference_tables",
    "lbracket_q4",
    "staircase_slit_q4",
    "mapped_q4",
    "q4_reference_tables",
    "read_abaqus_inp_quads",
    "read_gmsh_quads",
    "shape_functions",
    "shape_functions_3d",
    "shape_gradients",
    "shape_gradients_3d",
    "structured_hex8",
    "structured_q4",
]
