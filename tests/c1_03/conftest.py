from dataclasses import replace
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET
import pytest

from neurokinematics.kinematics.model import load_robot, RobotInputs
from neurokinematics.kinematics.torch_fk import TorchFK

ROOT = Path(__file__).resolve().parents[2]

def pytest_addoption(parser):
    parser.addoption('--mutation-output', default=None)

@pytest.fixture(scope='session')
def robot():
    return load_robot()

@pytest.fixture(scope='session')
def public_fk():
    return TorchFK.from_frozen()

def analytic_inputs(case):
    original = runpy.run_path(str(ROOT/'tests/f0_02/conftest.py'))['small_robot'].__wrapped__()
    if case in ('A1', 'A2', 'A3'):
        xml = ET.fromstring(original.urdf)
        if case == 'A2':
            ET.SubElement(xml.find("joint[@name='j1']"), 'origin', rpy='1.5707963267948966 0 0')
        if case == 'A3':
            xml.find("joint[@name='fixed_spacer']/parent").set('link', 'a')
            xml.find("joint[@name='j2']/parent").set('link', 'spacer')
            xml.find("joint[@name='tcp_joint']/parent").set('link', 'b')
        return replace(original, urdf=ET.tostring(xml))
    xml = b'''<robot name="analytic-one"><link name="world"/><link name="base"/><link name="arm"/><link name="tcp"/>
    <joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="3 4 5" rpy="0 0 1.5707963267948966"/></joint>
    <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="1 2 3"/><axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
    <joint name="tool" type="fixed"><parent link="arm"/><child link="tcp"/><origin xyz="2 0 0" rpy="1.5707963267948966 0 0"/></joint></robot>'''
    return RobotInputs(xml, ('j',), ((-4.,4.),), 'base', 'arm', 'tcp', 'analytic-one', {})
