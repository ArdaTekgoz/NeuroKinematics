# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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
        # Warn-only mode avoids hard crashes if an op has no deterministic kernel.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch versions may not support this API.
            pass
    else:
        # Default performant behavior
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
# Forward Kinematics (Position) — differentiable placeholder
# ============================================================


def forward_kinematics_position(theta: torch.Tensor) -> torch.Tensor:
    """
    Placeholder FK used in A-4.2 / A-4.3.2.

    Args:
        theta: (B, 6) joint angles

    Returns:
        ee_pos: (B, 3) end-effector position

    Why this exists:
    - Enforces FK-consistency between predicted joints and target Cartesian position.
    - Must be differentiable for FK-based training losses.

    NOTE:
    Replace with the true KR6 FK (e.g., DH chain) when moving beyond the synthetic baseline.
    """

    # Simple differentiable mapping (NOT physical KR6 FK).
    x = torch.sum(torch.cos(theta), dim=1)
    y = torch.sum(torch.sin(theta), dim=1)
    z = torch.sum(theta, dim=1) * 0.1
    return torch.stack([x, y, z], dim=1)


# ============================================================
# Losses (A-4.3.2)
# ============================================================


@dataclass(frozen=True)
class LossWeights:
    """
    Explicit, easy-to-modify loss weighting.

    A-4.3.2 effectively used:
      total = mse + fk_hard

    We keep both at weight=1.0 to preserve behavior.!
    """

    mse: float = 1.0
    fk: float = 1.0


@dataclass(frozen=True)
class FKHardSampleConfig:
    """
    Hard-sample mining configuration (A-4.3.2 style).

    weights = 1 + alpha * (error / mean_error)
    then clamp(weights, max=max_weight)

    - `mean_error` is detached: the weighting should not backprop through the batch statistic.
    - `eps` avoids division-by-zero when mean_error is very small.
    """

    alpha: float = 2.0
    max_weight: float = 5.0
    eps: float = 1e-6


def fk_hard_sample_position_loss(
    theta_pred: torch.Tensor,
    target_pos: torch.Tensor,
    cfg: FKHardSampleConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FK position loss with hard-sample weighting.

    Args:
        theta_pred: (B, 6)
        target_pos: (B, 3)
        cfg: weighting config

    Returns:
        loss: scalar tensor
        mean_err: scalar tensor (detached)
        max_err: scalar tensor (detached)
    """

    ee_pred = forward_kinematics_position(theta_pred)
    errors = torch.norm(ee_pred - target_pos, dim=1)  # (B,)

    # Detach mean: weights should respond to difficulty, not become another trainable pathway.
    mean_error = errors.mean().detach()
    denom = mean_error + cfg.eps
    weights = 1.0 + cfg.alpha * (errors / denom)
    weights = torch.clamp(weights, max=cfg.max_weight)

    loss = torch.mean(weights * errors)
    return loss, errors.mean().detach(), errors.max().detach()


# ============================================================
# Training / Validation
# ============================================================


@dataclass
class EpochStats:
    total: float = 0.0
    mse: float = 0.0
    fk: float = 0.0
    fk_mean: float = 0.0
    fk_max: float = 0.0
    n_batches: int = 0

    def update(
        self,
        total_loss: torch.Tensor,
        mse_loss: torch.Tensor,
        fk_loss: torch.Tensor,
        fk_mean: torch.Tensor,
        fk_max: torch.Tensor,
    ) -> None:
        self.total += float(total_loss.detach().cpu())
        self.mse += float(mse_loss.detach().cpu())
        self.fk += float(fk_loss.detach().cpu())
        self.fk_mean += float(fk_mean.detach().cpu())
        self.fk_max = max(self.fk_max, float(fk_max.detach().cpu()))
        self.n_batches += 1

    def averages(self) -> Dict[str, float]:
        denom = max(self.n_batches, 1)
        return {
            "total": self.total / denom,
            "mse": self.mse / denom,
            "fk": self.fk / denom,
            "fk_mean": self.fk_mean / denom,
            "fk_max": self.fk_max,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_criterion: nn.Module,
    device: torch.device,
    weights: LossWeights,
    fk_cfg: FKHardSampleConfig,
) -> EpochStats:
    model.train()
    stats = EpochStats()

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        theta_pred = model(x)
        mse_loss = mse_criterion(theta_pred, y)

        fk_loss, fk_mean, fk_max = fk_hard_sample_position_loss(
            theta_pred=theta_pred,
            target_pos=x[:, :3],
            cfg=fk_cfg,
        )

        total_loss = weights.mse * mse_loss + weights.fk * fk_loss

        total_loss.backward()
        optimizer.step()

        stats.update(
            total_loss=total_loss,
            mse_loss=mse_loss,
            fk_loss=fk_loss,
            fk_mean=fk_mean,
            fk_max=fk_max,
        )

    return stats


@torch.no_grad()
def validate_mse(
    model: nn.Module,
    loader: DataLoader,
    mse_criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validation behavior kept consistent with A-4.3.2:
    - Computes and prints only MSE on joint targets.

    (We do NOT add additional metrics by default to avoid changing "expected outputs".)
    """

    model.eval()
    val_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        theta_pred = model(x)
        loss = mse_criterion(theta_pred, y)
        val_loss += float(loss.detach().cpu())
        n_batches += 1

    return val_loss / max(n_batches, 1)


@torch.no_grad()
def validate_and_dump_fk_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dump_dir: str,
) -> None:
    """
    A-4.3.3 optional analysis:
    - Runs FK-based position error over the validation set.
    - Saves NumPy arrays for offline analysis.

    This does NOT modify training behavior or gradients.
    """

    model.eval()

    fk_errors_list = []
    ee_positions_list = []
    theta_pred_list = []

    for x, _ in loader:
        x = x.to(device)

        theta_pred = model(x)
        ee_pred = forward_kinematics_position(theta_pred)

        fk_error = torch.norm(ee_pred - x[:, :3], dim=1)  # (B,)

        fk_errors_list.append(fk_error)
        ee_positions_list.append(ee_pred)
        theta_pred_list.append(theta_pred)

    if not fk_errors_list:
        print("No validation samples available for FK error dump.")
        return

    fk_errors = torch.cat(fk_errors_list, dim=0).cpu().numpy()  # (N,)
    ee_positions = torch.cat(ee_positions_list, dim=0).cpu().numpy()  # (N, 3)
    theta_pred_all = torch.cat(theta_pred_list, dim=0).cpu().numpy()  # (N, 6)

    os.makedirs(dump_dir, exist_ok=True)

    np.save(os.path.join(dump_dir, "fk_errors.npy"), fk_errors)
    np.save(os.path.join(dump_dir, "ee_positions.npy"), ee_positions)
    np.save(os.path.join(dump_dir, "theta_pred.npy"), theta_pred_all)

    mean_err = float(np.mean(fk_errors))
    std_err = float(np.std(fk_errors))
    p95 = float(np.percentile(fk_errors, 95))
    p99 = float(np.percentile(fk_errors, 99))
    max_err = float(np.max(fk_errors))

    print(
        "FK Error Stats (validation) — A-4.3.3:"
        f" mean={mean_err:.6f},"
        f" std={std_err:.6f},"
        f" p95={p95:.6f},"
        f" p99={p99:.6f},"
        f" max={max_err:.6f}"
    )


# ============================================================
# CLI / Main
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train IKNet (Checkpoint A-4.3.2)")

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

    # Loss (A-4.3.2)
    p.add_argument("--loss_w_mse", type=float, default=1.0, help="Weight for joint MSE loss.")
    p.add_argument("--loss_w_fk", type=float, default=1.0, help="Weight for FK hard-sample loss.")
    p.add_argument("--fk_alpha", type=float, default=2.0, help="Hard-sample alpha (A-4.3.2: 2.0).")
    p.add_argument("--fk_max_weight", type=float, default=5.0, help="Hard-sample clamp (A-4.3.2: 5.0).")

    # Housekeeping (optional; does not change objectives)
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

    # A-4.3.3 optional validation-time FK error dumping
    p.add_argument(
        "--dump_validation_errors",
        action="store_true",
        help="If set, run FK error analysis on the validation set and dump NumPy arrays.",
    )
    p.add_argument(
        "--dump_dir",
        type=str,
        default="analysis_a_4_3_3",
        help="Output directory for A-4.3.3 validation FK error dumps.",
    )

    return p


def main() -> None:
    # Use parse_known_args() to ignore arguments passed by the Colab kernel.
    args, unknown = build_argparser().parse_known_args()

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

    # Model / Optimizer
    model = IKNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()

    # Loss configs (explicit)
    weights = LossWeights(mse=args.loss_w_mse, fk=args.loss_w_fk)
    fk_cfg = FKHardSampleConfig(alpha=args.fk_alpha, max_weight=args.fk_max_weight)

    # Training loop
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_criterion=mse_criterion,
            device=device,
            weights=weights,
            fk_cfg=fk_cfg,
        ).averages()

        # Epoch logging: same fields as A-4.3.2, but correctly averaged over epoch.
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"Total: {train_stats['total']:.4f} | "
            f"MSE: {train_stats['mse']:.4f} | "
            f"FK(mean): {train_stats['fk_mean']:.4f} | "
            f"FK(max): {train_stats['fk_max']:.4f}"
        )

    # Validation (kept consistent with A-4.3.2)
    val_mse = validate_mse(model=model, loader=val_loader, mse_criterion=mse_criterion, device=device)
    print("Validation MSE:", val_mse)

    # Optional A-4.3.3 validation-time FK error dump (no effect on training).
    if args.dump_validation_errors:
        validate_and_dump_fk_errors(
            model=model,
            loader=val_loader,
            device=device,
            dump_dir=args.dump_dir,
        )

    # Optional checkpoint save
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": "A-4.3.2",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            str(save_path),
        )
        print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()