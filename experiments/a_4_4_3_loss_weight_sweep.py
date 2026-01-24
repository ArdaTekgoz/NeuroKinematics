#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A-4.4.3 - Loss Weight Ablation Study (FK weight sweep)
=====================================================

Objective
---------
Systematically sweep FK loss weights (lambda_fk) to measure how FK supervision strength
affects IK learning quality, while keeping *everything else identical* to the
latest working training script `train_43.py` (A-4.3.3).

Hybrid loss:
    L_total = L_MSE + lambda_fk * L_FK

Important constraints (per task spec)
-------------------------------------
- ONLY lambda_fk changes between runs.
- Reuse the same: dataset, split, seed, model, optimizer, FK function,
  FK hard-sample loss formulation, logging style, and determinism setup.
- Self-contained script (do NOT refactor repo into modules).
- Output artifacts must follow the mandated directory structure.

Outputs (MANDATORY)
-------------------
experiments/
    a_4_4_3_loss_weight_sweep.py

analysis/
    a_4_4_3/
        lambda_0.0/
            fk_errors.npy
            summary.txt
            config.json
        lambda_0.1/
        lambda_0.5/
        lambda_1.0/
        lambda_2.0/
        lambda_5.0/
        aggregate_results.json
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# ============================================================
# Experiment Constants (FIXED per task spec)
# ============================================================

FK_WEIGHTS: List[float] = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

DATASET_PATH = Path("data") / "synthetic" / "KR6_FK_Dataset.csv"
EPOCHS = 100
BATCH_SIZE = 256
TRAIN_SPLIT = 0.9
SEED = 42
LR = 1e-3

ANALYSIS_ROOT = Path("analysis") / "a_4_4_3"


# ============================================================
# Reproducibility / Determinism (copied from train_43.py)
# ============================================================


def seed_everything(seed: int) -> None:
    """
    Best-effort reproducibility.

    Notes:
    - Full determinism across GPU kernels is not always possible/performance-friendly.
    - We keep the same behavior and do not introduce additional sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_determinism(deterministic: bool) -> None:
    """
    Toggle deterministic behavior where possible.
    (Same implementation as `train_43.py`.)
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
# Dataset (copied from train_43.py)
# ============================================================


class KR6IKDataset(Dataset):
    """
    CSV format (expected):
      Inputs : x, y, z, qx, qy, qz, qw
      Targets: t1, t2, t3, t4, t5, t6

    Column names intentionally match the A-4.3.x training scripts.
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
# Model (copied from train_43.py)
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
# Forward Kinematics (Position) - differentiable placeholder
# (copied from train_43.py)
# ============================================================


def forward_kinematics_position(theta: torch.Tensor) -> torch.Tensor:
    """
    Placeholder FK used in A-4.2 / A-4.3.x.

    Args:
        theta: (B, 6) joint angles

    Returns:
        ee_pos: (B, 3) end-effector position

    NOTE:
    Replace with the true KR6 FK (e.g., DH chain) when moving beyond the synthetic baseline.
    """
    x = torch.sum(torch.cos(theta), dim=1)
    y = torch.sum(torch.sin(theta), dim=1)
    z = torch.sum(theta, dim=1) * 0.1
    return torch.stack([x, y, z], dim=1)


# ============================================================
# Losses (A-4.3.x FK hard-sample loss; copied from train_43.py)
# ============================================================


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
    FK position loss with hard-sample weighting (A-4.3.x).

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
# Training (A-4.3.x style logging; only lambda_fk differs between runs)
# ============================================================


@dataclass
class EpochStats:
    total: float = 0.0
    mse: float = 0.0
    fk: float = 0.0
    n_batches: int = 0

    def update(self, total_loss: torch.Tensor, mse_loss: torch.Tensor, fk_loss: torch.Tensor) -> None:
        self.total += float(total_loss.detach().cpu())
        self.mse += float(mse_loss.detach().cpu())
        self.fk += float(fk_loss.detach().cpu())
        self.n_batches += 1

    def averages(self) -> Dict[str, float]:
        denom = max(self.n_batches, 1)
        return {
            "total": self.total / denom,
            "mse": self.mse / denom,
            "fk": self.fk / denom,
        }


def train_one_epoch_hybrid(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_criterion: nn.Module,
    device: torch.device,
    fk_cfg: FKHardSampleConfig,
    lambda_fk: float,
) -> EpochStats:
    """
    Train for one epoch with the hybrid loss:
        total = mse + lambda_fk * fk_hard

    Per spec:
    - FK loss is computed for every run (including lambda_fk=0.0).
    - For lambda_fk=0.0 it must NOT contribute to gradients; multiplying by 0.0
      guarantees the FK term has zero gradient contribution.
    """
    model.train()
    stats = EpochStats()

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        theta_pred = model(x)
        mse_loss = mse_criterion(theta_pred, y)

        fk_loss, _, _ = fk_hard_sample_position_loss(
            theta_pred=theta_pred,
            target_pos=x[:, :3],
            cfg=fk_cfg,
        )

        total_loss = mse_loss + float(lambda_fk) * fk_loss

        total_loss.backward()
        optimizer.step()

        stats.update(total_loss=total_loss, mse_loss=mse_loss, fk_loss=fk_loss)

    return stats


# ============================================================
# Validation / Analysis (per task spec)
# ============================================================


@torch.no_grad()
def evaluate_validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    mse_criterion: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute required validation metrics for a single run:
    - Joint-space MSE (mean over batches, consistent with existing scripts)
    - Raw FK errors (per-sample), where FK error is:
        || FK(theta_pred) - target_pos ||
      and target_pos comes from the dataset inputs (x,y,z).

    Returns:
        fk_errors_np: (N,) float64 NumPy array
        metrics: dict with fk_mean/std/p95/p99/max and joint_mse
    """
    model.eval()

    fk_errors: List[torch.Tensor] = []
    val_mse_total = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Validate", leave=False):
        x = x.to(device)
        y = y.to(device)

        theta_pred = model(x)

        # Joint-space MSE.
        mse_loss = mse_criterion(theta_pred, y)
        val_mse_total += float(mse_loss.detach().cpu())
        n_batches += 1

        # FK error vs target Cartesian position (xyz from input).
        ee_pred = forward_kinematics_position(theta_pred)
        errors = torch.norm(ee_pred - x[:, :3], dim=1)  # (B,)
        fk_errors.append(errors.detach().cpu())

    if not fk_errors:
        # Should never happen for non-empty datasets, but keep it safe.
        fk_errors_np = np.array([], dtype=np.float64)
    else:
        fk_errors_np = torch.cat(fk_errors, dim=0).numpy().astype(np.float64, copy=False)

    joint_mse = val_mse_total / max(n_batches, 1)

    # Percentiles: np.percentile on empty arrays will throw; handle the edge case.
    if fk_errors_np.size == 0:
        metrics = {
            "fk_mean": float("nan"),
            "fk_std": float("nan"),
            "fk_p95": float("nan"),
            "fk_p99": float("nan"),
            "fk_max": float("nan"),
            "joint_mse": float(joint_mse),
        }
        return fk_errors_np, metrics

    metrics = {
        "fk_mean": float(np.mean(fk_errors_np)),
        "fk_std": float(np.std(fk_errors_np)),
        "fk_p95": float(np.percentile(fk_errors_np, 95)),
        "fk_p99": float(np.percentile(fk_errors_np, 99)),
        "fk_max": float(np.max(fk_errors_np)),
        "joint_mse": float(joint_mse),
    }
    return fk_errors_np, metrics


def write_run_artifacts(
    output_dir: Path,
    fk_errors: np.ndarray,
    metrics: Dict[str, float],
    run_config: Dict,
) -> None:
    """
    Write the required per-run files:
    - fk_errors.npy
    - summary.txt
    - config.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "fk_errors.npy", fk_errors)

    summary_lines = [
        "A-4.4.3 - Loss Weight Ablation Study",
        f"lambda_fk: {run_config['lambda_fk']}",
        "",
        "Validation Metrics",
        f"FK mean error: {metrics['fk_mean']:.6f}",
        f"FK std error : {metrics['fk_std']:.6f}",
        f"FK p95 error : {metrics['fk_p95']:.6f}",
        f"FK p99 error : {metrics['fk_p99']:.6f}",
        f"FK max error : {metrics['fk_max']:.6f}",
        f"Joint-space MSE: {metrics['joint_mse']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


# ============================================================
# Main Sweep
# ============================================================


def main() -> None:
    # Device (fixed behavior; CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # We keep determinism "best effort" like train_43.py.
    # NOTE: The baselines use a CLI flag; here we keep it fixed to False unless
    # you edit the constant below. This avoids accidental changes between runs.
    deterministic = False
    num_workers = 0  # Windows-safe & reproducible default used across scripts.

    # Shared configs (must remain identical across runs)
    fk_cfg = FKHardSampleConfig(alpha=2.0, max_weight=5.0)

        # For reproducible split across all lambda runs, we always rebuild the dataset and split
    # under the same seed (so each run sees identical train/val indices).
    aggregate: Dict[str, Dict[str, float]] = {}

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    for lambda_fk in FK_WEIGHTS:
        # --------------------------------------------------------------------
        # Reproducibility reset per run (ensures same initialization per lambda)
        # --------------------------------------------------------------------
        seed_everything(SEED)
        configure_torch_determinism(deterministic)

        # --------------------------------------------------------------------
        # Dataset / Split (identical across runs)
        # --------------------------------------------------------------------
        dataset = KR6IKDataset(DATASET_PATH)

        if not (0.0 < TRAIN_SPLIT < 1.0):
            raise ValueError(f"TRAIN_SPLIT must be in (0, 1); got {TRAIN_SPLIT}")

        train_size = int(TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size

        g = torch.Generator()
        g.manual_seed(SEED)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # --------------------------------------------------------------------
        # Model / Optimizer (identical across runs)
        # --------------------------------------------------------------------
        model = IKNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        mse_criterion = nn.MSELoss()

        # --------------------------------------------------------------------
        # Training (ONLY lambda_fk differs)
        # --------------------------------------------------------------------
        print(f"\n=== A-4.4.3 Run: lambda_fk={lambda_fk} ===")
        for epoch in range(EPOCHS):
            stats = train_one_epoch_hybrid(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                mse_criterion=mse_criterion,
                device=device,
                fk_cfg=fk_cfg,
                lambda_fk=lambda_fk,
            ).averages()

            # Required logging: per-epoch Total loss, MSE loss, FK loss.
            print(
                f"[Epoch {epoch + 1:03d}] "
                f"Total: {stats['total']:.4f} | "
                f"MSE: {stats['mse']:.4f} | "
                f"FK: {stats['fk']:.4f}"
            )

        # --------------------------------------------------------------------
        # Validation Metrics + Artifact Dump
        # --------------------------------------------------------------------
        fk_errors_np, metrics = evaluate_validation_metrics(
            model=model,
            loader=val_loader,
            mse_criterion=mse_criterion,
            device=device,
        )

        run_dir = ANALYSIS_ROOT / f"lambda_{lambda_fk:.1f}"

        run_config = {
            "experiment": "A-4.4.3_loss_weight_sweep",
            "lambda_fk": float(lambda_fk),
            "fk_weights_sweep": FK_WEIGHTS,
            "dataset": str(DATASET_PATH),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_split": TRAIN_SPLIT,
            "seed": SEED,
            "lr": LR,
            "optimizer": "Adam",
            "model": "IKNet",
            "device": str(device),
            "deterministic": deterministic,
            "num_workers": num_workers,
            "fk_hard_sample_cfg": asdict(fk_cfg),
            "loss": "L_total = L_MSE + lambda_fk * L_FK (FK is hard-sample weighted position error)",
            "fk_error_definition": "|| FK(theta_pred) - target_pos || where target_pos = input xyz",
        }

        write_run_artifacts(
            output_dir=run_dir,
            fk_errors=fk_errors_np,
            metrics=metrics,
            run_config=run_config,
        )

        # One-line FK summary after each lambda run (required)
        print(
            f"[lambda_fk={lambda_fk}] "
            f"FK mean={metrics['fk_mean']:.6f}, "
            f"std={metrics['fk_std']:.6f}, "
            f"p95={metrics['fk_p95']:.6f}, "
            f"p99={metrics['fk_p99']:.6f}, "
            f"max={metrics['fk_max']:.6f} | "
            f"Joint-MSE={metrics['joint_mse']:.6f}"
        )

        # Stash for aggregate output (string keys for JSON stability)
        aggregate[f"{lambda_fk:.1f}"] = metrics

    # ------------------------------------------------------------------------
    # Aggregate Results (MANDATORY)
    # ------------------------------------------------------------------------
    aggregate_path = ANALYSIS_ROOT / "aggregate_results.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "A-4.4.3_loss_weight_sweep",
                "fk_weights": FK_WEIGHTS,
                "metrics_by_lambda_fk": aggregate,
            },
            f,
            indent=2,
        )

    # Required final log: print path to aggregate_results.json
    print(f"\nWrote aggregate results to: {aggregate_path}")


if __name__ == "__main__":
    main()

