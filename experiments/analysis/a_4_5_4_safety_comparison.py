import torch
import numpy as np
from inference.ik_inference_v4 import IKInferenceEngine

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

engine_unsafe = IKInferenceEngine(
    weights_path="checkpoints/a_4_5_2_base.pth",
    device="cpu",
    mode="unsafe"
)

engine_strict = IKInferenceEngine(
    weights_path="checkpoints/a_4_5_2_base.pth",
    device="cpu",
    mode="strict"
)

def random_pose():
    x = torch.randn(1,7).float()
    x[:, :3] /= torch.norm(x[:, :3])
    return x

def compute_gain(engine, x, eps):
    dx = torch.randn_like(x) * eps

    with torch.no_grad():
        q1 = engine.infer(x)
        q2 = engine.infer(x + dx)

    delta_q = torch.norm(q2 - q1).item()
    delta_x = torch.norm(dx).item()

    return delta_q / (delta_x + 1e-12)

# -----------------------------------------

N = 2000
eps = 1e-5

gains_unsafe = []
gains_strict = []

for _ in range(N):
    x = random_pose()

    gains_unsafe.append(compute_gain(engine_unsafe, x, eps))
    gains_strict.append(compute_gain(engine_strict, x, eps))

gains_unsafe = np.array(gains_unsafe)
gains_strict = np.array(gains_strict)

def report(name, gains):
    print(f"\n--- {name} ---")
    print("mean:", gains.mean())
    print("std:", gains.std())
    print("p95:", np.percentile(gains, 95))
    print("max:", gains.max())
    print("count(gain > 20):", np.sum(gains > 20))

report("UNSAFE", gains_unsafe)
report("STRICT", gains_strict)