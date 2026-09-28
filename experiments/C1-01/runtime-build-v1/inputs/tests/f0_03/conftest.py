import pytest

from neurokinematics.kinematics.model import RobotInputs, load_robot


def make_robot(joints, names, tip="flange", tcp="tcp"):
    """Temporary XML fixture. Every analytic test supplies its own expectations."""
    links = {"world", "base", tip, tcp}
    xml = []
    for name, kind, parent, child, xyz, rpy, axis in joints:
        links.update((parent, child))
        xml.append(f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
                   f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/>' +
                   (f'<axis xyz="{axis}"/><limit lower="-4" upper="4" effort="1" velocity="1"/>'
                    if kind == "revolute" else "") + '</joint>')
    payload = ('<robot name="analytic">' + ''.join(f'<link name="{n}"/>' for n in sorted(links))
               + ''.join(xml) + '</robot>').encode()
    return RobotInputs(payload, tuple(names), tuple((-4., 4.) for _ in names),
                       "base", tip, tcp, "analytic", {})


@pytest.fixture(scope="session")
def robot():
    return load_robot()


@pytest.fixture
def planar():
    # Fixed spacer BETWEEN active joints. L1=1+2=3, L2=1, TCP=1 -> 2.
    # Mount rotates world axes away from base and adds nonzero translation.
    return make_robot([
        ("mount", "fixed", "world", "base", "3 4 5", "0.4 -0.7 0.8", ""),
        ("j1", "revolute", "base", "a", "0 0 0", "0 0 0", "0 0 1"),
        ("spacer", "fixed", "a", "s", "2 0 0", "0 0 0", ""),
        ("j2", "revolute", "s", "b", "1 0 0", "0 0 0", "0 0 1"),
        ("tip", "fixed", "b", "flange", "1 0 0", "0 0 0", ""),
        ("tool", "fixed", "flange", "tcp", "1 0 0", "0.3 0.4 0.5", ""),
    ], ("j1", "j2"))
