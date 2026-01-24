# -*- coding: utf-8 -*-
"""
A-4.4.5.1: FK Outlier Suppression (Mean + 3σ Masking)

This experiment extends A-4.4.4 (curriculum FK weighting) by implementing
FK outlier detection and masking during training. FK outliers are defined
dynamically using batch-local statistics (mean + 3*std), and their FK loss
contributions are masked (ignored) while still maintaining gradient flow
for non-outliers.

Outlier Detection:
- Compute FK error per sample: e_i = ||FK(theta_pred) - target_pos||
- Batch statistics: fk_mean = mean(e), fk_std = std(e)
- Threshold: fk_threshold = fk_mean + 3.0 * fk_std
- Outlier: e_i > fk_threshold

Masking:
- mask_i = 1 if e_i <= fk_threshold else 0
- fk_loss = mean(mask * fk_loss_per_sample)
- Total loss: mse_loss + lambda_fk * fk_loss

Note: Samples are NOT dropped, only FK loss is masked. MSE loss is always applied.
"""

from __future__ import annotations

import argparse
import json
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
# Curriculum FK Weight Schedule
# ============================================================


def compute_curriculum_fk_weight(
    epoch: int, total_epochs: int, lambda_max: float, schedule_type: str = "linear"
) -> float:
    """
    Compute the FK loss weight for a given epoch using curriculum learning.

    Args:
        epoch: Current epoch index (0-indexed)
        total_epochs: Total number of training epochs
        lambda_max: Maximum FK weight to reach at the end of training
        schedule_type: Type of schedule ("linear" currently supported)

    Returns:
        Current lambda_fk value for this epoch

    Curriculum Schedule (linear):
        lambda_fk(epoch) = lambda_max * (epoch / (total_epochs - 1))
        - Epoch 0: lambda_fk = 0.0 (pure MSE learning)
        - Epoch (total_epochs-1): lambda_fk = lambda_max (full FK constraint)

    This gradually increases FK influence, allowing the model to first learn
    joint-space patterns, then refine with geometric constraints.
    """
    if total_epochs <= 1:
        return lambda_max

    if schedule_type == "linear":
        # Linear interpolation from 0 to lambda_max
        progress = epoch / (total_epochs - 1)
        return lambda_max * progress
    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}. Only 'linear' is implemented.")


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
# Losses with FK Outlier Detection and Masking
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


def compute_fk_errors_per_sample(
    theta_pred: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Compute FK error per sample (for outlier detection).

    Args:
        theta_pred: (B, 6) predicted joint angles
        target_pos: (B, 3) target end-effector position

    Returns:
        errors: (B,) FK error per sample
    """
    ee_pred = forward_kinematics_position(theta_pred)
    errors = torch.norm(ee_pred - target_pos, dim=1)  # (B,)
    return errors


def detect_fk_outliers(
    errors: torch.Tensor,
    sigma_multiplier: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect FK outliers using batch-local statistics (mean + 3σ).

    Args:
        errors: (B,) FK error per sample
        sigma_multiplier: Multiplier for standard deviation (default: 3.0)

    Returns:
        mask: (B,) binary mask (1 = inlier, 0 = outlier)
        threshold: scalar threshold value (detached)
    """
    fk_mean = errors.mean()
    fk_std = errors.std()

    # Safety check: if std == 0, no masking (all samples are inliers)
    if fk_std.item() == 0.0:
        mask = torch.ones_like(errors)
        threshold = fk_mean.detach()
        return mask, threshold

    fk_threshold = fk_mean + sigma_multiplier * fk_std

    # mask_i = 1 if e_i <= threshold else 0
    mask = (errors <= fk_threshold).float()

    return mask, fk_threshold.detach()


def fk_masked_position_loss(
    theta_pred: torch.Tensor,
    target_pos: torch.Tensor,
    cfg: FKHardSampleConfig,
    sigma_multiplier: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    FK position loss with outlier masking.

    Args:
        theta_pred: (B, 6)
        target_pos: (B, 3)
        cfg: hard-sample weighting config (used for non-outliers)
        sigma_multiplier: Multiplier for outlier detection (default: 3.0)

    Returns:
        loss: scalar tensor (masked FK loss)
        mean_err: scalar tensor (detached)
        max_err: scalar tensor (detached)
        mask: (B,) binary mask (1 = inlier, 0 = outlier)
        n_outliers: number of outliers in this batch
    """
    # Compute FK errors per sample
    errors = compute_fk_errors_per_sample(theta_pred, target_pos)  # (B,)

    # Detect outliers
    mask, threshold = detect_fk_outliers(errors, sigma_multiplier=sigma_multiplier)

    # Count outliers
    n_outliers = int((mask == 0).sum().item())

    # Apply hard-sample weighting to inliers only
    mean_error = errors.mean().detach()
    denom = mean_error + cfg.eps
    weights = 1.0 + cfg.alpha * (errors / denom)
    weights = torch.clamp(weights, max=cfg.max_weight)

    # Mask: only inliers contribute to loss
    masked_weights = mask * weights
    masked_errors = mask * errors

    # Compute masked loss: mean of (mask * weights * errors)
    # If all samples are outliers, loss is 0 (but this should be rare)
    loss = torch.mean(masked_weights * masked_errors)

    return loss, errors.mean().detach(), errors.max().detach(), mask, n_outliers


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
    n_outliers: int = 0
    n_samples: int = 0
    n_batches: int = 0

    def update(
        self,
        total_loss: torch.Tensor,
        mse_loss: torch.Tensor,
        fk_loss: torch.Tensor,
        fk_mean: torch.Tensor,
        fk_max: torch.Tensor,
        n_outliers: int,
        batch_size: int,
    ) -> None:
        self.total += float(total_loss.detach().cpu())
        self.mse += float(mse_loss.detach().cpu())
        self.fk += float(fk_loss.detach().cpu())
        self.fk_mean += float(fk_mean.detach().cpu())
        self.fk_max = max(self.fk_max, float(fk_max.detach().cpu()))
        self.n_outliers += n_outliers
        self.n_samples += batch_size
        self.n_batches += 1

    def averages(self) -> Dict[str, float]:
        denom = max(self.n_batches, 1)
        return {
            "total": self.total / denom,
            "mse": self.mse / denom,
            "fk": self.fk / denom,
            "fk_mean": self.fk_mean / denom,
            "fk_max": self.fk_max,
            "outliers": self.n_outliers,
            "total_samples": self.n_samples,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_criterion: nn.Module,
    device: torch.device,
    mse_weight: float,
    lambda_fk: float,  # Curriculum FK weight for this epoch
    fk_cfg: FKHardSampleConfig,
    sigma_multiplier: float = 3.0,
) -> EpochStats:
    """
    Train one epoch with curriculum FK weighting and outlier masking.

    Args:
        lambda_fk: Current FK loss weight (varies by epoch in curriculum learning)
        sigma_multiplier: Multiplier for outlier detection (default: 3.0)
    """
    model.train()
    stats = EpochStats()

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]

        optimizer.zero_grad(set_to_none=True)

        theta_pred = model(x)
        mse_loss = mse_criterion(theta_pred, y)

        # FK loss with outlier masking
        fk_loss, fk_mean, fk_max, mask, n_outliers = fk_masked_position_loss(
            theta_pred=theta_pred,
            target_pos=x[:, :3],
            cfg=fk_cfg,
            sigma_multiplier=sigma_multiplier,
        )

        # Curriculum FK weighting: lambda_fk increases over epochs
        total_loss = mse_weight * mse_loss + lambda_fk * fk_loss

        total_loss.backward()
        optimizer.step()

        stats.update(
            total_loss=total_loss,
            mse_loss=mse_loss,
            fk_loss=fk_loss,
            fk_mean=fk_mean,
            fk_max=fk_max,
            n_outliers=n_outliers,
            batch_size=batch_size,
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
    Validation behavior kept consistent with A-4.4.4:
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
) -> Dict[str, float]:
    """
    A-4.4.5.1 validation analysis (unchanged from A-4.4.4):
    - Runs FK-based position error over the validation set.
    - Saves NumPy arrays for offline analysis.
    - Returns statistics for summary.txt

    This does NOT modify training behavior or gradients.
    No masking in validation.
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
        return {}

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

    stats = {
        "mean": mean_err,
        "std": std_err,
        "p95": p95,
        "p99": p99,
        "max": max_err,
    }

    print(
        "FK Error Stats (validation) — A-4.4.5.1:"
        f" mean={mean_err:.6f},"
        f" std={std_err:.6f},"
        f" p95={p95:.6f},"
        f" p99={p99:.6f},"
        f" max={max_err:.6f}"
    )

    return stats


def save_summary_and_config(
    dump_dir: str,
    fk_stats: Dict[str, float],
    val_mse: float,
    args: argparse.Namespace,
    lambda_schedule: Dict[int, float],
) -> None:
    """
    Save summary.txt and config.json to the analysis directory.

    Args:
        dump_dir: Output directory
        fk_stats: FK error statistics from validation
        val_mse: Validation MSE loss
        args: Training arguments
        lambda_schedule: Dictionary mapping epoch -> lambda_fk for logging
    """
    os.makedirs(dump_dir, exist_ok=True)

    # Save summary.txt
    summary_path = os.path.join(dump_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("A-4.4.5.1 — FK Outlier Suppression (Mean + 3σ Masking) - Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Validation MSE: {val_mse:.6f}\n\n")
        if fk_stats:
            f.write("FK Error Statistics:\n")
            f.write(f"  Mean: {fk_stats['mean']:.6f}\n")
            f.write(f"  Std:  {fk_stats['std']:.6f}\n")
            f.write(f"  P95:  {fk_stats['p95']:.6f}\n")
            f.write(f"  P99:  {fk_stats['p99']:.6f}\n")
            f.write(f"  Max:  {fk_stats['max']:.6f}\n")
        f.write("\n")
        f.write("Curriculum Schedule:\n")
        f.write(f"  Schedule Type: {args.schedule_type}\n")
        f.write(f"  Lambda Max: {args.lambda_max}\n")
        f.write(f"  Total Epochs: {args.epochs}\n")
        f.write("\n")
        f.write("Outlier Detection:\n")
        f.write(f"  Method: Mean + 3σ (batch-local)\n")
        f.write(f"  Masking: FK loss masked for outliers\n")
        f.write("\n")
        f.write("Lambda FK Schedule (sample epochs):\n")
        # Show first, middle, and last epochs
        epochs_list = sorted(lambda_schedule.keys())
        if epochs_list:
            sample_epochs = [epochs_list[0]]
            if len(epochs_list) > 1:
                mid_idx = len(epochs_list) // 2
                sample_epochs.append(epochs_list[mid_idx])
            if len(epochs_list) > 2:
                sample_epochs.append(epochs_list[-1])
            for epoch in sample_epochs:
                f.write(f"  Epoch {epoch:03d}: lambda_fk = {lambda_schedule[epoch]:.6f}\n")

    # Save config.json
    config_path = os.path.join(dump_dir, "config.json")
    config_dict = {
        "experiment": "A-4.4.5.1",
        "description": "FK Outlier Suppression (Mean + 3σ Masking)",
        "args": vars(args),
        "lambda_schedule": lambda_schedule,
        "validation": {
            "mse": val_mse,
            "fk_stats": fk_stats,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved config to: {config_path}")


# ============================================================
# CLI / Main
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train IKNet with FK Outlier Masking (A-4.4.5.1)")

    # Data
    p.add_argument(
        "--dataset",
        type=str,
        default=str(Path("data/synthetic/KR6_FK_Dataset.csv")),
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

    # Loss (A-4.3.2 base, with curriculum FK)
    p.add_argument("--loss_w_mse", type=float, default=1.0, help="Weight for joint MSE loss.")
    p.add_argument(
        "--lambda_max",
        type=float,
        default=1.0,
        help="Maximum FK weight at end of training (curriculum schedule).",
    )
    p.add_argument(
        "--schedule_type",
        type=str,
        default="linear",
        choices=["linear"],
        help="Type of curriculum schedule (only 'linear' implemented).",
    )
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

    # A-4.4.5.1 validation output directory
    p.add_argument(
        "--dump_dir",
        type=str,
        default="analysis/a_4_4_5",
        help="Output directory for A-4.4.5.1 validation FK error dumps, summary.txt, and config.json.",
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
    fk_cfg = FKHardSampleConfig(alpha=args.fk_alpha, max_weight=args.fk_max_weight)

    # Track lambda_fk schedule for logging and config saving
    lambda_schedule: Dict[int, float] = {}

    # Outlier detection parameter (fixed at 3.0 as per requirements)
    sigma_multiplier = 3.0

    # Training loop with curriculum FK weighting and outlier masking
    print(f"\nA-4.4.5.1: FK Outlier Suppression (Mean + 3σ Masking)")
    print(f"  Schedule: {args.schedule_type}")
    print(f"  Lambda Max: {args.lambda_max}")
    print(f"  Total Epochs: {args.epochs}")
    print(f"  Outlier Detection: Mean + {sigma_multiplier}σ (batch-local)\n")

    for epoch in range(args.epochs):
        # Compute current lambda_fk using curriculum schedule
        lambda_fk = compute_curriculum_fk_weight(
            epoch=epoch,
            total_epochs=args.epochs,
            lambda_max=args.lambda_max,
            schedule_type=args.schedule_type,
        )
        lambda_schedule[epoch] = lambda_fk

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_criterion=mse_criterion,
            device=device,
            mse_weight=args.loss_w_mse,
            lambda_fk=lambda_fk,
            fk_cfg=fk_cfg,
            sigma_multiplier=sigma_multiplier,
        ).averages()

        # Epoch logging: includes current lambda_fk and outlier count
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"lambda_fk: {lambda_fk:.4f} | "
            f"MSE: {train_stats['mse']:.4f} | "
            f"FK(mean): {train_stats['fk_mean']:.4f} | "
            f"FK(max): {train_stats['fk_max']:.4f} | "
            f"Outliers: {train_stats['outliers']}/{train_stats['total_samples']} | "
            f"Total: {train_stats['total']:.4f}"
        )

    # Validation (kept consistent with A-4.4.4)
    val_mse = validate_mse(model=model, loader=val_loader, mse_criterion=mse_criterion, device=device)
    print(f"\nValidation MSE: {val_mse:.6f}")

    # A-4.4.5.1 validation-time FK error dump and summary/config saving (always performed)
    fk_stats = validate_and_dump_fk_errors(
        model=model,
        loader=val_loader,
        device=device,
        dump_dir=args.dump_dir,
    )
    save_summary_and_config(
        dump_dir=args.dump_dir,
        fk_stats=fk_stats,
        val_mse=val_mse,
        args=args,
        lambda_schedule=lambda_schedule,
    )

    # Optional checkpoint save
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": "A-4.4.5.1",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "lambda_schedule": lambda_schedule,
            },
            str(save_path),
        )
        print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
