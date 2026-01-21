# -*- coding: utf-8 -*-
"""
A-4.4.2 ablation: FK-only inverse kinematics training.

This script is intentionally identical to A-4.4.1 (`a_4_4_1_mse_only.py`)
in data/model/optimizer/seeding/determinism/logging, **except**:
    - Training loss uses ONLY forward-kinematics position error.
    - No joint-space MSE supervision; ground-truth joint angles are ignored.

Rationale:
    Testing whether IK can be learned purely from end-effector positions,
    without direct joint supervision (ablation vs MSE-only and hybrid setups).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# ============================================================
# Reproducibility / Determinism
# ============================================================


def seed_everything(seed: int) -> None:
    """
    Best-effort reproducibility.

    Notes:
    - Full determinism across GPU kernels is not always possible/performance-friendly.
    - We expose `--deterministic` for stricter behavior when needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_determinism(deterministic: bool) -> None:
    """
    Toggle deterministic behavior where possible.
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch versions may not support this API.
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================
# Dataset
# ============================================================


class KR6IKDataset(Dataset):
    """
    CSV format (expected):
      Inputs : x, y, z, qx, qy, qz, qw
      Targets: t1, t2, t3, t4, t5, t6

    We keep the exact column names used in A-4.3.2.
    """

    INPUT_COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    TARGET_COLUMNS = ["t1", "t2", "t3", "t4", "t5", "t6"]

    def __init__(self, csv_path: str | os.PathLike):
        csv_path = str(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        missing_x = [c for c in self.INPUT_COLUMNS if c not in df.columns]
        missing_y = [c for c in self.TARGET_COLUMNS if c not in df.columns]
        if missing_x or missing_y:
            raise ValueError(
                "Dataset CSV is missing required columns.\n"
                f"  Missing inputs : {missing_x}\n"
                f"  Missing targets: {missing_y}\n"
                f"  Available columns: {list(df.columns)}"
            )

        x = df[self.INPUT_COLUMNS].values.astype(np.float32, copy=False)
        y = df[self.TARGET_COLUMNS].values.astype(np.float32, copy=False)

        self.X = torch.from_numpy(x)
        self.Y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ============================================================
# Model
# ============================================================


class IKNet(nn.Module):
    """
    MLP architecture from A-4.3.2 (kept intentionally for checkpoint parity).
    """

    def __init__(self, in_dim: int = 7, out_dim: int = 6):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Forward Kinematics (Position)
# ============================================================


def forward_kinematics_position(theta: torch.Tensor) -> torch.Tensor:
    """
    Placeholder FK used in A-4.2 / A-4.3.2.

    Unchanged: FK function must remain identical across ablations.
    """
    x = torch.sum(torch.cos(theta), dim=1)
    y = torch.sum(torch.sin(theta), dim=1)
    z = torch.sum(theta, dim=1) * 0.1
    return torch.stack([x, y, z], dim=1)


# ============================================================
# Training (FK-only loss — no joint supervision)
# ============================================================


def train_one_epoch_fk_only(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single-epoch training using ONLY FK position error.

    We deliberately ignore ground-truth joint angles to test whether the model
    can recover IK solely from end-effector positions (ablation vs MSE-only/hybrid).
    """
    model.train()
    total_fk = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        # y is intentionally unused (no joint-space supervision in this ablation).
        x = x.to(device)

        # Target end-effector position is taken from input pose (xyz).
        target_pos = x[:, :3]

        optimizer.zero_grad(set_to_none=True)
        theta_pred = model(x)
        ee_pred = forward_kinematics_position(theta_pred)

        # FK-only loss: mean L2 distance between predicted EE and target EE position.
        fk_loss = torch.mean(torch.norm(ee_pred - target_pos, dim=1))

        fk_loss.backward()
        optimizer.step()

        total_fk += float(fk_loss.detach().cpu())
        n_batches += 1

    return total_fk / max(n_batches, 1)


# ============================================================
# Validation (FK metrics only; no gradients)
# ============================================================


@torch.no_grad()
def evaluate_fk_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Compute FK-based position error on the validation set.
    This matches A-4.4.1 evaluation for paper-quality comparability.
    """
    model.eval()
    fk_errors = []

    for x, y in tqdm(loader, desc="Validate-FK", leave=False):
        x = x.to(device)
        y = y.to(device)

        # Same evaluation protocol: compare predicted FK vs GT joints -> FK.
        theta_pred = model(x)
        ee_pred = forward_kinematics_position(theta_pred)
        ee_gt = forward_kinematics_position(y)
        errors = torch.norm(ee_pred - ee_gt, dim=1)  # (B,)
        fk_errors.append(errors)

    if not fk_errors:
        print("No validation samples available for FK error evaluation.")
        return {}

    fk_errors_tensor = torch.cat(fk_errors, dim=0)
    fk_errors_np = fk_errors_tensor.cpu().numpy()

    metrics = {
        "fk_mean": float(np.mean(fk_errors_np)),
        "fk_std": float(np.std(fk_errors_np)),
        "fk_p95": float(np.percentile(fk_errors_np, 95)),
        "fk_p99": float(np.percentile(fk_errors_np, 99)),
        "fk_max": float(np.max(fk_errors_np)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "fk_errors.npy", fk_errors_np)

    summary_lines = [
        "A-4.4.2 FK-only Training (no joint supervision)",
        f"FK mean: {metrics['fk_mean']:.6f}",
        f"FK std : {metrics['fk_std']:.6f}",
        f"FK p95 : {metrics['fk_p95']:.6f}",
        f"FK p99 : {metrics['fk_p99']:.6f}",
        f"FK max : {metrics['fk_max']:.6f}",
    ]
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(
        "FK Error Stats (validation — A-4.4.2 FK-only): "
        f"mean={metrics['fk_mean']:.6f}, "
        f"std={metrics['fk_std']:.6f}, "
        f"p95={metrics['fk_p95']:.6f}, "
        f"p99={metrics['fk_p99']:.6f}, "
        f"max={metrics['fk_max']:.6f}"
    )

    return metrics


# ============================================================
# CLI / Main
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A-4.4.2 FK-only Training (no joint supervision)")

    # Data
    p.add_argument(
        "--dataset",
        type=str,
        default=str(Path("/content/") / "KR6_FK_Dataset.csv"),
        help="Path to CSV dataset (default: data/synthetic/KR6_FK_Dataset.csv)",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_split", type=float, default=0.9, help="Train fraction (rest is val).")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable best-effort deterministic training (may be slower).",
    )

    # Housekeeping
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 for Windows safety/reproducibility.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path to save model checkpoint (.pt). If empty, no checkpoint is saved.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="analysis/a_4_4_2",
        help="Where to store FK validation artifacts.",
    )

    return p


def main() -> None:
    # Use parse_known_args() to ignore arguments passed by the Colab kernel.
    args, _ = build_argparser().parse_known_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Reproducibility
    seed_everything(args.seed)
    configure_torch_determinism(args.deterministic)

    # Dataset + split
    dataset = KR6IKDataset(args.dataset)

    if not (0.0 < args.train_split < 1.0):
        raise ValueError(f"--train_split must be in (0, 1); got {args.train_split}")

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Use a generator so `random_split` is reproducible across runs.
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model / Optimizer (identical to A-4.4.1 to isolate the loss change).
    model = IKNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop (FK-only loss).
    for epoch in range(args.epochs):
        train_fk = train_one_epoch_fk_only(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # Required training log: FK-only end-effector loss (no joint MSE).
        print(f"[Epoch {epoch + 1:03d}] Train FK-pos loss: {train_fk:.4f} (A-4.4.2)")

    # Validation FK analysis (no gradients, same protocol as A-4.4.1).
    output_dir = Path(args.output_dir)
    metrics = evaluate_fk_validation(
        model=model,
        loader=val_loader,
        device=device,
        output_dir=output_dir,
    )

    # Persist minimal config for traceability.
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model": "IKNet_A4_4_2_FK_only",
        "loss": "FK-only position (no joint supervision)",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Optional checkpoint save (kept for parity).
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": "A-4.4.2_FK_only",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "fk_metrics": metrics,
            },
            str(save_path),
        )
        print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()


