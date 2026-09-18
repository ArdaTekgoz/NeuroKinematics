import pytest

from neurokinematics.kinematics.model import RobotInputs, load_robot


@pytest.fixture(scope="session")
def robot():
    return load_robot()


@pytest.fixture
def small_robot():
    # Analytic planar arm: j1, length 1, j2, fixed spacer length 2,
    # TCP length 1 plus +90deg about X. XML order intentionally scrambled.
    xml = b'''<robot name="analytic">
    <link name="world"/><link name="base"/><link name="a"/>
    <link name="b"/><link name="spacer"/><link name="tcp"/>
    <joint name="j2" type="revolute"><parent link="a"/><child link="b"/>
      <origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
    <joint name="mount" type="fixed"><parent link="world"/><child link="base"/>
      <origin xyz="3 4 5" rpy="0 0 1.5707963267948966"/></joint>
    <joint name="tcp_joint" type="fixed"><parent link="spacer"/><child link="tcp"/>
      <origin xyz="1 0 0" rpy="1.5707963267948966 0 0"/></joint>
    <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
      <axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
    <joint name="fixed_spacer" type="fixed"><parent link="b"/><child link="spacer"/>
      <origin xyz="2 0 0"/></joint></robot>'''
    return RobotInputs(xml, ("j1", "j2"), ((-4., 4.), (-4., 4.)), "base", "spacer", "tcp", "analytic", {})
