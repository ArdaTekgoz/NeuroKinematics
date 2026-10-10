import math
import pytest
import torch
from neurokinematics.neural.c106r_pose_followup import pose_objective


@pytest.mark.parametrize('position,angle,expected',[(0,0,0),(.002,0,1),(0,math.radians(1),1),(.002,math.radians(1),2)])
def test_profile_scaled_objective(position,angle,expected):
    t=torch.eye(4,dtype=torch.float64)
    t[0,3]=position
    t[:3,:3]=torch.tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]],dtype=torch.float64)
    loss=pose_objective(t,torch.zeros(3,dtype=torch.float64),torch.eye(3,dtype=torch.float64))
    assert float(loss)==pytest.approx(expected,abs=1e-12)
