import torch
import torch.nn as nn


# ============================================================
# Train.py ile birebir uyumlu model
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
# Inference Engine
# ============================================================

class IKInferenceEngine:
    def __init__(
        self,
        weights_path: str,
        device: str = None,
        joint_limits=None,
    ):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = IKNet().to(self.device)
        self.model.eval()

        if weights_path is None:
            raise ValueError("weights_path must be provided")

        state = torch.load(weights_path, map_location=self.device)

        if "model_state_dict" not in state:
            raise ValueError("Checkpoint missing 'model_state_dict'")

        self.model.load_state_dict(state["model_state_dict"])

        self.joint_limits = joint_limits

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    def _validate_input(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be torch.Tensor")

        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError(f"Expected input shape (B,7), got {x.shape}")

        if torch.isnan(x).any():
            raise ValueError("Input contains NaN")

        return x.float()

    # --------------------------------------------------------
    # Joint limits
    # --------------------------------------------------------

    def _apply_joint_limits(self, joints: torch.Tensor):

        if self.joint_limits is None:
            return joints

        for i, (jmin, jmax) in enumerate(self.joint_limits):
            joints[:, i] = torch.clamp(joints[:, i], jmin, jmax)

        return joints

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def infer(self, x: torch.Tensor):

        x = self._validate_input(x)
        x = x.to(self.device)

        joints = self.model(x)
        joints = self._apply_joint_limits(joints)

        return joints.cpu()
