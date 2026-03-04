import torch
import numpy as np
from inference.ik_inference_v4 import IKInferenceEngine

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

def compute_gain(x, eps):
    dx = torch.randn_like(x) * eps

    with torch.no_grad():
        q1 = engine.infer(x)
        q2 = engine.infer(x + dx)

    delta_q = torch.norm(q2 - q1).item()
    delta_x = torch.norm(dx).item()

    return delta_q / (delta_x + 1e-12)

# -------------------------------------------------------
# MAIN TEST
# -------------------------------------------------------

N = 2000
eps_levels = [1e-6, 1e-5, 1e-4, 1e-3]

for eps in eps_levels:
    gains = []

    for _ in range(N):
        x = random_pose()
        gain = compute_gain(x, eps)
        gains.append(gain)

    gains = np.array(gains)

    print(f"\n--- EPS = {eps} ---")
    print("mean gain:", gains.mean())
    print("std gain:", gains.std())
    print("p95 gain:", np.percentile(gains, 95))
    print("max gain:", gains.max())