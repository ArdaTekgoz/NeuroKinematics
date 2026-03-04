import torch
import numpy as np
import pandas as pd
import math

from core.dh_fk import forward_kinematics
from core.robot_kr6 import KR6_DH_PARAMS

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Joint limits (realistic industrial range)
# ----------------------------

JOINT_LIMITS = [
    (-2.97, 2.97),
    (-2.09, 2.09),
    (-2.97, 2.97),
    (-2.09, 2.09),
    (-2.97, 2.97),
    (-2.09, 2.09),
]

def sample_random_joints(n):
    joints = []
    for low, high in JOINT_LIMITS:
        joints.append(
            torch.rand(n) * (high - low) + low
        )
    return torch.stack(joints, dim=1)

# ----------------------------
# Generate dataset
# ----------------------------

N = 100000  # sample count

q = sample_random_joints(N)

with torch.no_grad():
    pose = forward_kinematics(q, KR6_DH_PARAMS)

pose = pose.numpy()
q = q.numpy()

data = np.concatenate([pose, q], axis=1)

columns = [
    "x","y","z",
    "qx","qy","qz","qw",
    "t1","t2","t3","t4","t5","t6"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("KR6_3D_Dataset.csv", index=False)

print("Dataset generated.")
print("Shape:", df.shape)
print(df.describe())