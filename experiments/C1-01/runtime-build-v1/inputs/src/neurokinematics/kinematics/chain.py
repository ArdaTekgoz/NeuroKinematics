"""Independent ElementTree URDF parser, restricted to fixed/revolute tree joints."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .transforms import origin_transform, vector3


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray | None
    limits: tuple[float, float] | None


def _vector(element, attribute, default=None):
    text = None if element is None else element.get(attribute)
    if text is None:
        if default is None:
            raise ValueError(f"missing {attribute}")
        return vector3(default)
    return vector3([float(v) for v in text.split()])


def extract_chain(urdf: bytes, base: str, tcp: str, joint_names) -> tuple[Joint, ...]:
    """Reject malformed/ambiguous graphs; retain fixed joints only on base->TCP path.

    Missing origin components use URDF's identity defaults. Revolute axis/limits
    are required explicitly by this project's narrower input contract.
    """
    root = ET.fromstring(urdf)
    if root.tag != "robot":
        raise ValueError("URDF root must be robot")
    link_list = [link.get("name") for link in root.findall("link")]
    links = set(link_list)
    if None in links or "" in links or len(links) != len(link_list):
        raise ValueError("missing or duplicate link names")
    if base not in links or tcp not in links:
        raise ValueError("base/TCP frame missing")
    by_child = {}
    names = set()
    for node in root.findall("joint"):
        name, kind = node.get("name"), node.get("type")
        if not name or name in names:
            raise ValueError("missing or duplicate joint name")
        names.add(name)
        if kind not in {"fixed", "revolute"} or node.find("mimic") is not None:
            raise ValueError(f"unsupported joint: {name} ({kind})")
        parent_node, child_node = node.find("parent"), node.find("child")
        parent = None if parent_node is None else parent_node.get("link")
        child = None if child_node is None else child_node.get("link")
        if parent not in links or child not in links:
            raise ValueError(f"missing parent/child link: {name}")
        if child in by_child:
            raise ValueError(f"multiple parents / ambiguous chain: {child}")
        origin = node.find("origin")
        transform = origin_transform(_vector(origin, "xyz", (0, 0, 0)),
                                     _vector(origin, "rpy", (0, 0, 0)))
        axis, limits = None, None
        if kind == "revolute":
            axis = _vector(node.find("axis"), "xyz")
            if not np.isclose(np.linalg.norm(axis), 1, atol=1e-12, rtol=0):
                raise ValueError(f"axis must be unit length: {name}")
            limit = node.find("limit")
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError(f"missing limits: {name}")
            limits = (float(limit.get("lower")), float(limit.get("upper")))
            if not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f"invalid limits: {name}")
            axis.setflags(write=False)
        transform.setflags(write=False)
        by_child[child] = Joint(name, kind, parent, child, transform, axis, limits)
    # Check every component, including side branches, for cycles/disconnection.
    for link in links:
        visited = set()
        while link in by_child:
            if link in visited:
                raise ValueError("cyclic URDF graph")
            visited.add(link)
            link = by_child[link].parent
    if len(links - by_child.keys()) != 1:
        raise ValueError("URDF must be a connected tree")
    path = []
    cursor = tcp
    while cursor != base:
        if cursor not in by_child:
            raise ValueError("no directed base-to-TCP chain")
        joint = by_child[cursor]
        path.append(joint)
        cursor = joint.parent
    path.reverse()
    active = [j.name for j in path if j.kind == "revolute"]
    if len(set(joint_names)) != len(joint_names) or set(active) != set(joint_names):
        raise ValueError("manifest active joints differ from base-to-TCP chain")
    # The tree can have fixed side branches, but no unaccounted active mechanism.
    if {j.name for j in by_child.values() if j.kind == "revolute"} != set(active):
        raise ValueError("active joints outside the serial chain")
    return tuple(path)
