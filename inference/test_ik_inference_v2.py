import torch
from ik_inference_v2 import IKInferenceEngine

joint_limits = [
    (-3.14, 3.14),
    (-1.57, 1.57),
    (-2.0, 2.0),
    (-3.14, 3.14),
    (-2.5, 2.5),
    (-6.28, 6.28),
]

engine = IKInferenceEngine(
    joint_limits=joint_limits,
    conf_threshold=0.5
)

x = torch.randn(4,7)

joints, conf, mask = engine.infer(x)

print("Joints:", joints.shape)
print("Confidence:", conf.shape if conf is not None else None)
print("Mask:", mask.shape if mask is not None else None)
