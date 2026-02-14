import torch
import argparse
from pathlib import Path
from typing import Tuple, Optional

from core.model import IKNet


class IKInferenceEngine:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        predict_confidence: bool = True,
        conf_threshold: float = 0.5,
    ):
        self.device = self._select_device(device)
        self.conf_threshold = conf_threshold

        self.model = IKNet(predict_confidence=predict_confidence)
        self.model.to(self.device)
        self.model.eval()

        if weights_path:
            self._load_weights(weights_path)

        torch.set_grad_enabled(False)

    def _select_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_weights(self, weights_path: str):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def infer(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: shape (N, 7)
        returns:
            joints: (N, 6)
            confidence: (N, 1) or None
        """
        x = x.to(self.device)

        outputs = self.model(x)

        if isinstance(outputs, tuple):
            joints, conf = outputs

            if conf is not None:
                mask = conf >= self.conf_threshold
                joints = joints * mask

            return joints.cpu(), conf.cpu()

        return outputs.cpu(), None

    def export_torchscript(self, export_path: str):
        dummy_input = torch.randn(1, 7).to(self.device)
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(export_path)
        print(f"TorchScript model exported to {export_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    args = parser.parse_args()

    engine = IKInferenceEngine(
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf_threshold,
    )

    x = torch.randn(args.batch, 7)
    joints, conf = engine.infer(x)

    print("Input shape:", x.shape)
    print("Joint output shape:", joints.shape)
    if conf is not None:
        print("Confidence shape:", conf.shape)


if __name__ == "__main__":
    main()
