import torch
import numpy as np
from inference.ik_inference_v4 import IKInferenceEngine
from kinematics.fk import forward_kinematics  # <-- kendi FK fonksiyonun

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

engine = IKInferenceEngine(
    weights_path="checkpoints/a_4_5_2_base.pth",
    device="cpu",
    mode="strict"
)

def random_pose():
    x = torch.randn(1,7).float()
    x[:, :3] /= torch.norm(x[:, :3])
    return x

def quat_angle_error(q1, q2):
    dot = torch.sum(q1 * q2, dim=1).clamp(-1.0, 1.0)
    return 2 * torch.acos(torch.abs(dot))

# -------------------------------------------------

N = 2000

pos_errors = []
rot_errors = []

for _ in range(N):

    x_target = random_pose()

    with torch.no_grad():
        q_pred = engine.infer(x_target)

    x_recon = forward_kinematics(q_pred)

    # Position error
    pos_err = torch.norm(
        x_target[:, :3] - x_recon[:, :3],
        dim=1
    ).item()

    # Orientation error (radian)
    rot_err = quat_angle_error(
        x_target[:, 3:],
        x_recon[:, 3:]
    ).item()

    pos_errors.append(pos_err)
    rot_errors.append(rot_err)

pos_errors = np.array(pos_errors)
rot_errors = np.array(rot_errors)

print("\n--- POSITION ERROR (meters) ---")
print("mean:", pos_errors.mean())
print("std:", pos_errors.std())
print("p95:", np.percentile(pos_errors, 95))
print("max:", pos_errors.max())

print("\n--- ORIENTATION ERROR (radians) ---")
print("mean:", rot_errors.mean())
print("std:", rot_errors.std())
print("p95:", np.percentile(rot_errors, 95))
print("max:", rot_errors.max())