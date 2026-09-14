"""Readers for ASCII Gmsh 2.2 quadrilateral and Abaqus CPS4/CPS4R meshes.

Node labels are compacted to zero-based indices and clockwise elements are
rewound counter-clockwise."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _section(lines: Sequence[str], name: str) -> List[str]:
    start_marker = "$" + name
    end_marker = "$End" + name
    try:
        start = lines.index(start_marker)
        end = lines.index(end_marker, start + 1)
    except ValueError as error:
        raise ValueError("Gmsh file is missing {} section".format(name)) from error
    if end <= start + 1:
        raise ValueError("Gmsh {} section is empty".format(name))
    return list(lines[start + 1 : end])


def _compact_connectivity(
    node_ids: Sequence[int], element_node_ids: Sequence[Sequence[int]]
) -> np.ndarray:
    node_index: Dict[int, int] = {}
    for index, node_id in enumerate(node_ids):
        if node_id in node_index:
            raise ValueError("duplicate node id {}".format(node_id))
        node_index[node_id] = index
    connectivity = np.empty((len(element_node_ids), 4), dtype=np.int64)
    for element, labels in enumerate(element_node_ids):
        for local, label in enumerate(labels):
            if label not in node_index:
                raise ValueError("element references undefined node id {}".format(label))
            connectivity[element, local] = node_index[label]
    return connectivity


def _repair_ccw(coords: np.ndarray, econn: np.ndarray) -> np.ndarray:
    if econn.ndim != 2 or econn.shape[1] != 4:
        raise ValueError("quad connectivity must have shape (Ne, 4)")
    points = coords[econn]
    x = points[..., 0]
    y = points[..., 1]
    signed_twice_area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    if np.any(~np.isfinite(signed_twice_area)):
        raise ValueError("quad element has non-finite signed area")
    if np.any(signed_twice_area == 0.0):
        element = int(np.flatnonzero(signed_twice_area == 0.0)[0])
        raise ValueError("quad element {} has zero signed area".format(element))
    repaired = econn.copy()
    flipped = signed_twice_area < 0.0
    repaired[flipped] = repaired[flipped][:, [0, 3, 2, 1]]
    return repaired


def read_gmsh_quads(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read nodes and first-order quads from a Gmsh 2.2 ASCII ``.msh`` file.

    Only element type 3 (four-node quadrilateral) is accepted.  Node labels
    may be non-contiguous; returned connectivity uses zero-based compact row
    indices in file node order.
    """
    source = Path(path)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]

    mesh_format = _section(lines, "MeshFormat")
    if len(mesh_format) != 1:
        raise ValueError("Gmsh MeshFormat section must contain one data line")
    fields = mesh_format[0].split()
    if len(fields) != 3 or fields[0] != "2.2" or fields[1] != "0":
        raise ValueError("only Gmsh 2.2 ASCII files are supported")

    node_section = _section(lines, "Nodes")
    try:
        node_count = int(node_section[0])
    except (ValueError, IndexError) as error:
        raise ValueError("invalid Gmsh node count") from error
    if node_count <= 0 or len(node_section) != node_count + 1:
        raise ValueError("Gmsh Nodes section count does not match its records")
    node_ids: List[int] = []
    coords = np.empty((node_count, 2), dtype=np.float64)
    for index, line in enumerate(node_section[1:]):
        values = line.split()
        if len(values) != 4:
            raise ValueError("Gmsh node records must contain id, x, y, z")
        try:
            node_ids.append(int(values[0]))
            coords[index] = [float(values[1]), float(values[2])]
            z = float(values[3])
        except ValueError as error:
            raise ValueError("invalid Gmsh node record: {}".format(line)) from error
        if not np.isfinite(z) or not np.all(np.isfinite(coords[index])):
            raise ValueError("Gmsh node coordinates must be finite")

    element_section = _section(lines, "Elements")
    try:
        element_count = int(element_section[0])
    except (ValueError, IndexError) as error:
        raise ValueError("invalid Gmsh element count") from error
    if element_count <= 0 or len(element_section) != element_count + 1:
        raise ValueError("Gmsh Elements section count does not match its records")
    element_nodes: List[List[int]] = []
    element_ids = set()
    for line in element_section[1:]:
        values = line.split()
        if len(values) < 3:
            raise ValueError("invalid Gmsh element record: {}".format(line))
        try:
            element_id = int(values[0])
            element_type = int(values[1])
            n_tags = int(values[2])
        except ValueError as error:
            raise ValueError("invalid Gmsh element record: {}".format(line)) from error
        if element_id in element_ids:
            raise ValueError("duplicate element id {}".format(element_id))
        element_ids.add(element_id)
        if element_type != 3:
            raise ValueError(
                "unsupported Gmsh element type {}; only 4-node quads (type 3) are supported".format(
                    element_type
                )
            )
        if n_tags < 0 or len(values) != 3 + n_tags + 4:
            raise ValueError("Gmsh quad record must contain exactly four node ids")
        try:
            element_nodes.append([int(value) for value in values[-4:]])
        except ValueError as error:
            raise ValueError("invalid Gmsh quad node id") from error

    connectivity = _compact_connectivity(node_ids, element_nodes)
    return coords, _repair_ccw(coords, connectivity)


def read_abaqus_inp_quads(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``*NODE`` and ``*ELEMENT, TYPE=CPS4(R)`` from an Abaqus input."""
    source = Path(path)
    node_ids: List[int] = []
    node_coords: List[List[float]] = []
    element_nodes: List[List[int]] = []
    element_ids = set()
    mode = None
    saw_supported_element_section = False

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [part.strip() for part in line.split(",")]
            keyword = parts[0].upper()
            if keyword == "*NODE":
                mode = "node"
            elif keyword == "*ELEMENT":
                options = {}
                for part in parts[1:]:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        options[key.strip().upper()] = value.strip().upper()
                element_type = options.get("TYPE")
                if element_type not in ("CPS4", "CPS4R"):
                    raise ValueError(
                        "unsupported Abaqus element type {!r}; only CPS4 and "
                        "CPS4R are supported".format(element_type)
                    )
                mode = "element"
                saw_supported_element_section = True
            else:
                mode = None
            continue

        values = [value.strip() for value in line.split(",")]
        if mode == "node":
            if len(values) < 3:
                raise ValueError("Abaqus node records require id, x, y")
            try:
                node_ids.append(int(values[0]))
                xy = [float(values[1]), float(values[2])]
            except ValueError as error:
                raise ValueError("invalid Abaqus node record: {}".format(line)) from error
            if not np.all(np.isfinite(xy)):
                raise ValueError("Abaqus node coordinates must be finite")
            node_coords.append(xy)
        elif mode == "element":
            if len(values) != 5:
                raise ValueError("Abaqus CPS4(R) records require id and four node ids")
            try:
                element_id = int(values[0])
                labels = [int(value) for value in values[1:]]
            except ValueError as error:
                raise ValueError("invalid Abaqus element record: {}".format(line)) from error
            if element_id in element_ids:
                raise ValueError("duplicate element id {}".format(element_id))
            element_ids.add(element_id)
            element_nodes.append(labels)

    if not node_ids:
        raise ValueError("Abaqus input contains no *NODE records")
    if not saw_supported_element_section or not element_nodes:
        raise ValueError("Abaqus input contains no CPS4(R) elements")
    coords = np.asarray(node_coords, dtype=np.float64)
    connectivity = _compact_connectivity(node_ids, element_nodes)
    return coords, _repair_ccw(coords, connectivity)
