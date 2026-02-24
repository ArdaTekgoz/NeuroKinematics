import torch
import torch.nn as nn


# ============================================================
# IKNet (Train.py compatible)
# ============================================================

class IKNet(nn.Module):
    def __init__(self, in_dim=7, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Production-Grade Inference Engine (Dual Mode)
# ============================================================

class IKInferenceEngine:
    def __init__(
        self,
        weights_path: str,
        device: str = None,
        joint_limits=None,
        mode: str = "strict",          # "strict" or "safe"
        max_output_abs: float = 1e3,   # safety threshold
    ):

        if mode not in ("strict", "safe"):
            raise ValueError("mode must be 'strict' or 'safe'")

        self.mode = mode
        self.max_output_abs = max_output_abs
        self.joint_limits = joint_limits

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = IKNet().to(self.device)
        self.model.eval()

        # Hard-disable training behaviour
        for module in self.model.modules():
            if hasattr(module, "training"):
                module.training = False

        if weights_path is None:
            raise ValueError("weights_path must be provided")

        checkpoint = torch.load(weights_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Invalid checkpoint format")

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"'model_state_dict' missing. Keys: {list(checkpoint.keys())}"
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    def _validate_input(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be torch.Tensor")

        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError(f"Expected input shape (B,7), got {x.shape}")

        if self.mode == "strict":
            if torch.isnan(x).any():
                raise RuntimeError("Safety violation: NaN detected in input")

            if torch.isinf(x).any():
                raise RuntimeError("Safety violation: Inf detected in input")

        else:  # SAFE MODE
            x = x.clone()
            x[torch.isnan(x)] = 0.0
            x[torch.isinf(x)] = 0.0
            x = torch.clamp(x, -1e6, 1e6)

        return x

    # --------------------------------------------------------
    # Joint Limit Enforcement
    # --------------------------------------------------------

    def _enforce_joint_limits(self, joints: torch.Tensor):

        if self.joint_limits is None:
            return joints

        lower = torch.tensor(
            [jl[0] for jl in self.joint_limits],
            device=joints.device,
            dtype=joints.dtype,
        )

        upper = torch.tensor(
            [jl[1] for jl in self.joint_limits],
            device=joints.device,
            dtype=joints.dtype,
        )

        if self.mode == "strict":
            if (joints < lower).any() or (joints > upper).any():
                raise RuntimeError("Safety violation: joint limit exceeded")
            return joints

        else:  # SAFE
            return torch.max(torch.min(joints, upper), lower)

    # --------------------------------------------------------
    # Output Validation
    # --------------------------------------------------------

    def _validate_output(self, joints: torch.Tensor):

        if torch.isnan(joints).any():
            if self.mode == "strict":
                raise RuntimeError("Safety violation: NaN detected in output")
            else:
                joints = torch.nan_to_num(joints, nan=0.0)

        if torch.isinf(joints).any():
            if self.mode == "strict":
                raise RuntimeError("Safety violation: Inf detected in output")
            else:
                joints = torch.nan_to_num(joints, posinf=0.0, neginf=0.0)

        max_abs = torch.abs(joints).max()

        if max_abs > self.max_output_abs:
            if self.mode == "strict":
                raise RuntimeError(
                    "Safety violation: output magnitude exceeded safety threshold"
                )
            else:
                joints = torch.clamp(
                    joints,
                    -self.max_output_abs,
                    self.max_output_abs,
                )

        return joints

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def infer(self, x: torch.Tensor):

        x = self._validate_input(x)
        x = x.to(self.device, dtype=torch.float32)

        joints = self.model(x)

        joints = self._validate_output(joints)
        joints = self._enforce_joint_limits(joints)

        return joints.cpu()