"""
NeuroKinematics — Robot Model (URDF Parser)

Parses URDF files to extract the kinematic chain: joints, links,
joint types, axes, limits, and parent-child relationships.
"""

import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class JointInfo:
    """Stores parsed information for a single URDF joint."""
    name: str
    joint_type: str              # 'revolute', 'prismatic', 'fixed', 'continuous'
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray       # Translation (3,)
    origin_rpy: np.ndarray       # Roll-Pitch-Yaw (3,)
    axis: np.ndarray             # Joint axis (3,)
    limit_lower: float = -np.pi
    limit_upper: float = np.pi
    limit_effort: float = 0.0
    limit_velocity: float = 0.0


@dataclass
class LinkInfo:
    """Stores parsed information for a single URDF link."""
    name: str


class RobotModel:
    """
    Robot kinematic model parsed from a URDF file.

    Extracts the full kinematic chain (link-joint tree), joint types,
    axes, limits, and constructs the ordered active joint list from
    base_link to the TCP (tool center point).

    Usage:
        model = RobotModel.from_urdf("robots/kuka_kr6/kr6.urdf")
        print(model.n_joints)       # 6
        print(model.joint_names)    # ['joint_a1', ..., 'joint_a6']
        print(model.joint_limits)   # (array([...]), array([...]))
    """

    def __init__(self, robot_name: str, joints: List[JointInfo], links: List[LinkInfo]):
        self.robot_name = robot_name
        self._all_joints = joints
        self._all_links = links

        # Build link-joint tree
        self._link_map: Dict[str, LinkInfo] = {l.name: l for l in links}
        self._joint_map: Dict[str, JointInfo] = {j.name: j for j in joints}
        self._parent_to_joints: Dict[str, List[JointInfo]] = {}
        for j in joints:
            self._parent_to_joints.setdefault(j.parent_link, []).append(j)

        # Discover base link (link that is never a child)
        child_links = {j.child_link for j in joints}
        all_link_names = {l.name for l in links}
        base_candidates = all_link_names - child_links
        self.base_link = sorted(base_candidates)[0] if base_candidates else links[0].name

        # Build ordered kinematic chain (active joints only: revolute + prismatic)
        self._chain_joints: List[JointInfo] = []
        self._chain_links: List[str] = [self.base_link]
        self._build_chain(self.base_link)

        # TCP is the last child link in the chain
        self.tcp_link = self._chain_links[-1] if self._chain_links else self.base_link

    def _build_chain(self, current_link: str):
        """Recursively traverses the kinematic tree to build the ordered joint chain."""
        if current_link not in self._parent_to_joints:
            return
        for joint in self._parent_to_joints[current_link]:
            if joint.joint_type in ('revolute', 'prismatic', 'continuous'):
                self._chain_joints.append(joint)
            # Also follow fixed joints to reach TCP
            self._chain_links.append(joint.child_link)
            self._build_chain(joint.child_link)

    @classmethod
    def from_urdf(cls, urdf_path: str) -> 'RobotModel':
        """
        Parses a URDF XML file and constructs a RobotModel.

        Args:
            urdf_path: Path to the .urdf file.

        Returns:
            RobotModel instance.
        """
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        tree = ET.parse(str(path))
        root = tree.getroot()
        robot_name = root.attrib.get('name', 'unknown_robot')

        # Parse links
        links = []
        for link_elem in root.findall('link'):
            links.append(LinkInfo(name=link_elem.attrib['name']))

        # Parse joints
        joints = []
        for joint_elem in root.findall('joint'):
            name = joint_elem.attrib['name']
            joint_type = joint_elem.attrib.get('type', 'fixed')

            parent_link = joint_elem.find('parent').attrib['link']
            child_link = joint_elem.find('child').attrib['link']

            # Origin
            origin_elem = joint_elem.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.attrib.get('xyz', '0 0 0')
                rpy_str = origin_elem.attrib.get('rpy', '0 0 0')
                origin_xyz = np.array([float(v) for v in xyz_str.split()])
                origin_rpy = np.array([float(v) for v in rpy_str.split()])
            else:
                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)

            # Axis
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                axis_str = axis_elem.attrib.get('xyz', '0 0 1')
                axis = np.array([float(v) for v in axis_str.split()])
            else:
                axis = np.array([0.0, 0.0, 1.0])

            # Limits
            limit_lower = -np.pi
            limit_upper = np.pi
            limit_effort = 0.0
            limit_velocity = 0.0
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                limit_lower = float(limit_elem.attrib.get('lower', -np.pi))
                limit_upper = float(limit_elem.attrib.get('upper', np.pi))
                limit_effort = float(limit_elem.attrib.get('effort', 0.0))
                limit_velocity = float(limit_elem.attrib.get('velocity', 0.0))

            joints.append(JointInfo(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                limit_effort=limit_effort,
                limit_velocity=limit_velocity,
            ))

        return cls(robot_name=robot_name, joints=joints, links=links)

    # ======================== Properties ========================

    @property
    def n_joints(self) -> int:
        """Number of active (non-fixed) joints in the kinematic chain."""
        return len(self._chain_joints)

    @property
    def joint_names(self) -> List[str]:
        """Ordered list of active joint names."""
        return [j.name for j in self._chain_joints]

    @property
    def joint_types(self) -> List[str]:
        """Ordered list of active joint types."""
        return [j.joint_type for j in self._chain_joints]

    @property
    def joint_axes(self) -> np.ndarray:
        """Joint axes as (n_joints, 3) array."""
        return np.array([j.axis for j in self._chain_joints])

    @property
    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_limits, upper_limits) arrays of shape (n_joints,)."""
        lower = np.array([j.limit_lower for j in self._chain_joints])
        upper = np.array([j.limit_upper for j in self._chain_joints])
        return lower, upper

    @property
    def chain_joints(self) -> List[JointInfo]:
        """Ordered list of JointInfo objects in the kinematic chain."""
        return list(self._chain_joints)

    @property
    def all_joints(self) -> List[JointInfo]:
        """All joints in the URDF, including fixed joints."""
        return list(self._all_joints)

    def __repr__(self):
        return (f"RobotModel(name='{self.robot_name}', "
                f"n_joints={self.n_joints}, "
                f"base='{self.base_link}', "
                f"tcp='{self.tcp_link}')")
