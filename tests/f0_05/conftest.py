import numpy as np
import pytest

from neurokinematics.kinematics.model import RobotInputs, load_robot


@pytest.fixture(scope="session")
def robot():
    return load_robot()


@pytest.fixture
def analytic_robot():
    def build(count=1):
        links = ['world', 'base'] + [f'a{i}' for i in range(count)] + ['tcp']
        joints = ['<joint name="mount" type="fixed"><parent link="world"/><child link="base"/>'
                  '<origin xyz="3 4 5" rpy="0.4 -0.7 0.8"/></joint>']
        for i in range(count):
            parent = 'base' if i == 0 else f'a{i-1}'
            joints.append(f'<joint name="j{i}" type="revolute"><parent link="{parent}"/>'
                          f'<child link="a{i}"/><origin xyz="{0 if i == 0 else 1} 0 0"/>'
                          '<axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>')
        joints.append(f'<joint name="tool" type="fixed"><parent link="a{count-1}"/>'
                      '<child link="tcp"/><origin xyz="1 0 0" rpy="0.3 0.4 0.5"/></joint>')
        xml = ('<robot name="analytic">'+''.join(f'<link name="{name}"/>' for name in links)
               + ''.join(joints)+'</robot>').encode()
        return RobotInputs(xml, tuple(f'j{i}' for i in range(count)), ((-4., 4.),)*count,
                           'base', f'a{count-1}', 'tcp', 'analytic', {})
    return build


@pytest.fixture
def real_pose(robot):
    from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
    from neurokinematics.data.factory import canonical_quaternion
    q = np.array([.3, -.6, .8, -1., .5, -.7])
    transform = PinocchioFK(robot).reference_forward_kinematics(q)
    return q, transform[:3, 3], canonical_quaternion(transform[:3, :3])
