import torch
import torch.nn as nn
from model import IKNet


class IKInferenceEngine:
    def __init__(
        self,
        weights_path=None,
        device=None,
        conf_threshold=0.5,
        joint_limits=None,
        normalize=False,
        mean=None,
        std=None,
        strict_input=True,
        safe_mode=False,
        max_position_norm=2.5,
        quat_tolerance=0.05,
    ):

        # Device
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model
        self.model = IKNet(predict_confidence=True).to(self.device)
        self.model.eval()

        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.conf_threshold = conf_threshold
        self.joint_limits = joint_limits

        # Normalization (train tarafında yok, default False)
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor(mean).to(self.device)
            self.std = torch.tensor(std).to(self.device)

        # Input safety config
        self.strict_input = strict_input
        self.safe_mode = safe_mode
        self.max_position_norm = max_position_norm
        self.quat_tolerance = quat_tolerance

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    def _validate_input_shape(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be torch.Tensor")

        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError("Input must have shape (B,7)")

        if torch.isnan(x).any():
            raise ValueError("Input contains NaN")

        if torch.isinf(x).any():
            raise ValueError("Input contains Inf")

        return x.float()

    def _apply_normalization(self, x):
        return (x - self.mean) / self.std

    # ------------------------------------------------------------------
    # Safety Guards
    # ------------------------------------------------------------------

    def _guard_position(self, x):
        pos = x[:, :3]
        pos_norm = torch.norm(pos, dim=1)

        invalid = pos_norm > self.max_position_norm

        if invalid.any():
            if self.strict_input:
                raise ValueError(
                    f"Position norm exceeds limit ({self.max_position_norm})."
                )
            elif self.safe_mode:
                scale = self.max_position_norm / (pos_norm + 1e-8)
                scale = torch.clamp(scale, max=1.0)
                x[:, :3] = pos * scale.unsqueeze(1)

        return x

    def _guard_quaternion(self, x):
        quat = x[:, 3:]
        quat_norm = torch.norm(quat, dim=1)

        deviation = torch.abs(quat_norm - 1.0)
        invalid = deviation > self.quat_tolerance

        if invalid.any():
            if self.strict_input:
                raise ValueError(
                    f"Quaternion norm deviation exceeds tolerance ({self.quat_tolerance})."
                )
            elif self.safe_mode:
                quat = quat / (quat_norm.unsqueeze(1) + 1e-8)
                x[:, 3:] = quat

        return x

    # ------------------------------------------------------------------
    # Joint Limits
    # ------------------------------------------------------------------

    def _apply_joint_limits(self, joints):
        if self.joint_limits is None:
            return joints

        for i, (jmin, jmax) in enumerate(self.joint_limits):
            joints[:, i] = torch.clamp(joints[:, i], jmin, jmax)

        return joints

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, x):

        x = self._validate_input_shape(x)
        x = x.to(self.device)

        # Safety guards
        x = self._guard_position(x)
        x = self._guard_quaternion(x)

        # Optional normalization
        if self.normalize:
            x = self._apply_normalization(x)

        # Forward pass
        joints, conf = self.model(x)

        # Joint safety
        joints = self._apply_joint_limits(joints)

        mask = None
        if conf is not None:
            mask = conf >= self.conf_threshold

        return (
            joints.cpu(),
            conf.cpu() if conf is not None else None,
            mask.cpu() if mask is not None else None,
        )