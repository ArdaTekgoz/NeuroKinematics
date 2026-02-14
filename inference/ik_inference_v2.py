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
    ):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = IKNet(predict_confidence=True).to(self.device)
        self.model.eval()

        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.conf_threshold = conf_threshold

        # Joint limits
        # joint_limits = [(min,max), ...] length=6
        self.joint_limits = joint_limits

        # Normalization
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor(mean).to(self.device)
            self.std = torch.tensor(std).to(self.device)

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be torch.Tensor")

        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError("Input must have shape (B,7)")

        if torch.isnan(x).any():
            raise ValueError("Input contains NaN")

        return x.float()

    def _apply_normalization(self, x):
        return (x - self.mean) / self.std

    def _apply_joint_limits(self, joints):
        if self.joint_limits is None:
            return joints

        for i, (jmin, jmax) in enumerate(self.joint_limits):
            joints[:, i] = torch.clamp(joints[:, i], jmin, jmax)

        return joints

    @torch.no_grad()
    def infer(self, x):

        x = self._validate_input(x)
        x = x.to(self.device)

        if self.normalize:
            x = self._apply_normalization(x)

        joints, conf = self.model(x)

        joints = self._apply_joint_limits(joints)

        mask = None
        if conf is not None:
            mask = conf >= self.conf_threshold

        return joints.cpu(), conf.cpu(), mask.cpu() if mask is not None else None
