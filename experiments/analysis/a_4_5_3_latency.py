import torch
import numpy as np
import time
from inference.ik_inference_v4 import IKInferenceEngine

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

engine = IKInferenceEngine(
    weights_path="checkpoints/a_4_5_2_base.pth",
    device="cpu",
    mode="strict"
)

pose = torch.randn(1,7).float()
pose[:, :3] /= torch.norm(pose[:, :3])

# -------------------------------------------------
# Warm-up
# -------------------------------------------------

for _ in range(100):
    engine.infer(pose)

# -------------------------------------------------
# Full Pipeline Latency
# -------------------------------------------------

N = 5000
times = []

for _ in range(N):
    start = time.perf_counter()
    engine.infer(pose)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

times = np.array(times)

print("\n--- Full Pipeline Latency (ms) ---")
print("min:", times.min())
print("mean:", times.mean())
print("std:", times.std())
print("p95:", np.percentile(times, 95))
print("max:", times.max())

# -------------------------------------------------
# Pure Forward Latency
# -------------------------------------------------

model = engine.model
model.eval()

times_forward = []

with torch.no_grad():
    for _ in range(N):
        start = time.perf_counter()
        model(pose)
        end = time.perf_counter()
        times_forward.append((end - start) * 1000)

times_forward = np.array(times_forward)

print("\n--- Forward Only Latency (ms) ---")
print("min:", times_forward.min())
print("mean:", times_forward.mean())
print("std:", times_forward.std())
print("p95:", np.percentile(times_forward, 95))
print("max:", times_forward.max())