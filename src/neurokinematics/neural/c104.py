"""Frozen C1-04 data, MLP, training, checkpoint and validation contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from neurokinematics.data.factory import read_shard
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


EVIDENCE = ROOT / "experiments/C1-04"
CONFIG = EVIDENCE / "config.json"
SCHEMA = EVIDENCE / "checkpoint-schema.json"
INPUTS = EVIDENCE / "input-hashes.json"
NORMALIZATION = ROOT / "experiments/C1-02/normalization.json"
PAIR_SCHEMA = ROOT / "experiments/C1-02/schema.json"
PAIR_MANIFEST = ROOT / "experiments/C1-02/dataset-manifest.json"
DATA_ROOT = ROOT / "data/generated/C1-02/v1"
VARIANTS = ("pose_only", "conditioned")
FIELDS = ("position_m", "quaternion_wxyz", "q_current")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def contracts() -> tuple[dict, dict, dict]:
    config, schema, norm = read_json(CONFIG), read_json(SCHEMA), read_json(NORMALIZATION)
    if config["source"]["dataset_content_sha256"] != read_json(PAIR_MANIFEST)["dataset_content_sha256"]:
        raise ValueError("C1-02 data identity drift")
    if config["features"]["pose_only"] != ["position_m[0:3]", "quaternion_wxyz[0:4]"]:
        raise ValueError("pose feature order drift")
    if config["features"]["conditioned"] != ["position_m[0:3]", "quaternion_wxyz[0:4]", "q_current[0:6]"]:
        raise ValueError("conditioned feature order drift")
    if norm["source"] != "C1-02_train_only" or norm["position_count"] != 16800 or norm["label_count"] != 15204:
        raise ValueError("C1-02 normalization provenance drift")
    if np.any(np.asarray(norm["position_std_m"], dtype=np.float64) <= 0):
        raise ValueError("invalid position std")
    return config, schema, norm


@dataclass(frozen=True)
class Rows:
    split: str
    pair_id: np.ndarray
    source_sample_id: np.ndarray
    group_id: np.ndarray
    family: np.ndarray
    mode: np.ndarray
    label_present: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    q_current: np.ndarray
    q_target: np.ndarray
    pose_only: np.ndarray
    conditioned: np.ndarray
    target_normalized: np.ndarray

    def take(self, indices: np.ndarray) -> "Rows":
        return Rows(self.split, *(getattr(self, name)[indices] for name in self.__dataclass_fields__ if name != "split"))

    def feature(self, variant: str) -> np.ndarray:
        if variant not in VARIANTS:
            raise ValueError("unknown variant")
        return getattr(self, variant)


def _decode(a: np.ndarray) -> np.ndarray:
    return np.char.decode(a, "ascii")


def _read_split(split: str, config: dict, norm: dict, *, label_fk: bool) -> Rows:
    if split not in ("train", "validation"):
        raise ValueError("test and benchmark are sealed for C1-04")
    manifest, schema = read_json(PAIR_MANIFEST), read_json(PAIR_SCHEMA)
    fields = schema["fields"]
    order = [field["name"] for field in fields]
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in order}
    for shard in manifest["shards"]:
        if f"-{split}-" not in shard["path"]:
            continue
        path = DATA_ROOT / shard["path"]
        if sha(path) != shard["file_sha256"]:
            raise ValueError(f"shard SHA drift: {path}")
        arrays = read_shard(path, order)
        for field in fields:
            name = field["name"]
            a = arrays[name]
            if a.dtype.str != np.dtype(field["dtype"]).str or a.shape != (shard["record_count"], *field["shape_per_record"]):
                raise ValueError(f"shard dtype/shape drift: {name}")
            buckets[name].append(a)
    data = {name: np.concatenate(parts) for name, parts in buckets.items()}
    if len(data["pair_id"]) != (16800 if split == "train" else 3600):
        raise ValueError("split count drift")
    if not np.all(_decode(data["split"]) == split):
        raise ValueError("split mismatch")
    pair_id = _decode(data["pair_id"])
    if len(set(pair_id.tolist())) != len(pair_id):
        raise ValueError("duplicate pair id")
    ordered = np.argsort(pair_id, kind="stable")
    data = {name: value[ordered] for name, value in data.items()}
    pair_id = pair_id[ordered]
    present = data["label_present"].astype(bool)
    mode = _decode(data["pair_mode"])
    if not np.all(np.isin(mode, ("local", "wide"))):
        raise ValueError("bad pair mode")
    if np.any(~present & (mode != "wide")):
        raise ValueError("missing local label")
    if int(present.sum()) != (15204 if split == "train" else 3249):
        raise ValueError("label count drift")
    p = data["position_m"]
    quat = data["quaternion_wxyz"]
    current = data["q_current"]
    target = data["q_target"]
    if not all(np.isfinite(x).all() for x in (p, quat, current)):
        raise ValueError("nonfinite model input")
    if np.any(np.abs(np.linalg.norm(quat, axis=1) - 1) > 1e-12) or np.any(quat[:, 0] < 0):
        raise ValueError("noncanonical quaternion")
    if np.any(~np.isfinite(target[present])) or not np.isnan(target[~present]).all():
        raise ValueError("bad missing label sentinel")
    inputs = load_robot()
    lower, upper = np.asarray(inputs.limits, dtype=np.float64).T
    if np.any(current < lower) or np.any(current > upper) or np.any(target[present] < lower) or np.any(target[present] > upper):
        raise ValueError("joint limit violation")
    if label_fk:
        fk = PinocchioFK(inputs)
        for index in np.flatnonzero(present):
            t = fk.reference_forward_kinematics(target[index])
            pe = np.linalg.norm(t[:3, 3] - p[index])
            re = rotation_error(t[:3, :3], quaternion_rotation(quat[index]))
            if pe > 0.001 or re > math.radians(0.5):
                raise ValueError(f"label/target FK mismatch: {pair_id[index]}")
    pos = (p - np.asarray(norm["position_mean_m"])) / np.asarray(norm["position_std_m"])
    qscaled = (current - lower) / (upper - lower)
    label = np.zeros_like(target)
    label[present] = (target[present] - lower) / (upper - lower)
    pose_only = np.concatenate((pos, quat), axis=1).astype(np.float32)
    conditioned = np.concatenate((pos, quat, qscaled), axis=1).astype(np.float32)
    if pose_only.shape[1] != 7 or conditioned.shape[1] != 13 or not np.isfinite(conditioned).all():
        raise ValueError("feature width/nonfinite")
    rows = Rows(split, pair_id, _decode(data["source_sample_id"]), _decode(data["group_id"]),
                _decode(data["source_family"]), mode, present, p, quat, current, target,
                pose_only, conditioned, label.astype(np.float32))
    validate_rows(rows)
    return rows


def validate_rows(rows: Rows) -> None:
    if rows.split not in ("train", "validation"):
        raise ValueError("sealed split")
    n = len(rows.pair_id)
    if any(len(getattr(rows, name)) != n for name in rows.__dataclass_fields__ if name != "split"):
        raise ValueError("row length mismatch")
    if len(set(rows.pair_id.tolist())) != n or np.any(~np.isin(rows.mode, ("local", "wide"))):
        raise ValueError("pair identity or mode mismatch")
    if np.any(~rows.label_present & (rows.mode != "wide")):
        raise ValueError("missing local label")
    if rows.pose_only.shape != (n, 7) or rows.conditioned.shape != (n, 13) or rows.target_normalized.shape != (n, 6):
        raise ValueError("feature or target width mismatch")
    if not all(np.isfinite(x).all() for x in (rows.position, rows.quaternion, rows.q_current,
                                              rows.pose_only, rows.conditioned, rows.target_normalized)):
        raise ValueError("nonfinite input or target")
    if np.any(~np.isfinite(rows.q_target[rows.label_present])) or not np.isnan(rows.q_target[~rows.label_present]).all():
        raise ValueError("label sentinel mismatch")
    if np.any(rows.target_normalized[~rows.label_present] != 0):
        raise ValueError("missing label entered supervised target")
    if np.any(np.abs(np.linalg.norm(rows.quaternion, axis=1) - 1) > 1e-12) or np.any(rows.quaternion[:, 0] < 0):
        raise ValueError("noncanonical quaternion")
    lower, upper = np.asarray(load_robot().limits, dtype=np.float64).T
    if np.any(rows.q_current < lower) or np.any(rows.q_current > upper) or \
            np.any(rows.q_target[rows.label_present] < lower) or np.any(rows.q_target[rows.label_present] > upper):
        raise ValueError("joint limit violation")
    config, _, norm = contracts()
    pos = (rows.position - np.asarray(norm["position_mean_m"])) / np.asarray(norm["position_std_m"])
    expected_pose = np.concatenate((pos, rows.quaternion), axis=1).astype(np.float32)
    expected_conditioned = np.concatenate((pos, rows.quaternion,
                                          (rows.q_current - lower) / (upper - lower)), axis=1).astype(np.float32)
    if not np.array_equal(rows.pose_only, expected_pose) or not np.array_equal(rows.conditioned, expected_conditioned):
        raise ValueError("feature order/scaling mismatch")
    expected_target = ((rows.q_target[rows.label_present] - lower) / (upper - lower)).astype(np.float32)
    if not np.array_equal(rows.target_normalized[rows.label_present], expected_target):
        raise ValueError("target scaling mismatch")


def validate_split_pair(train: Rows, validation: Rows) -> None:
    validate_rows(train)
    validate_rows(validation)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("wrong split for training/monitor")
    if set(train.group_id.tolist()) & set(validation.group_id.tolist()):
        raise ValueError("train/validation group leakage")
    if set(train.source_sample_id.tolist()) & set(validation.source_sample_id.tolist()):
        raise ValueError("train/validation root leakage")


def load_data(*, label_fk: bool = True) -> tuple[Rows, Rows]:
    config, _, norm = contracts()
    train = _read_split("train", config, norm, label_fk=label_fk)
    validation = _read_split("validation", config, norm, label_fk=label_fk)
    validate_split_pair(train, validation)
    return train, validation


def controlled_subset(train: Rows, validation: Rows) -> tuple[Rows, Rows]:
    ti = np.flatnonzero((train.family == "main") & (train.mode == "local") & train.label_present)[:64]
    vi = np.flatnonzero((validation.family == "main") & (validation.mode == "local") & validation.label_present)[:32]
    if len(ti) != 64 or len(vi) != 32:
        raise ValueError("pilot subset missing")
    return train.take(ti), validation.take(vi)


def reject_shifted_labels(rows: Rows) -> dict:
    if len(rows.pair_id) != 64 or len(set(rows.source_sample_id.tolist())) != 64:
        raise ValueError("negative control requires 64 distinct roots")
    shifted = np.roll(rows.q_target, 1, axis=0)
    fk = PinocchioFK(load_robot())
    failures = []
    for index, q in enumerate(shifted):
        t = fk.reference_forward_kinematics(q)
        pe = float(np.linalg.norm(t[:3, 3] - rows.position[index]))
        re = float(math.degrees(rotation_error(t[:3, :3], quaternion_rotation(rows.quaternion[index]))))
        if pe > 0.001 or re > 0.5:
            failures.append({"pair_id": str(rows.pair_id[index]), "position_m": pe, "orientation_deg": re})
    if not failures:
        raise ValueError("shifted-label negative control was not rejected")
    return {"status": "PASS_REJECTED", "shifted_rows": 64, "rejected_rows": len(failures), "first_failure": failures[0]}


class MLP(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown variant")
        width = 7 if variant == "pose_only" else 13
        self.layers = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                    nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 6))
        expected = read_json(CONFIG)["models"]["parameter_count"][variant]
        if sum(p.numel() for p in self.parameters()) != expected:
            raise ValueError("parameter count mismatch")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def setup(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _rss_bytes() -> int:
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes
    class Counter(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counter()
    counters.cb = ctypes.sizeof(Counter)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counter), wintypes.DWORD)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _metrics(model: MLP, rows: Rows, variant: str, *, batch: int = 1024) -> dict:
    selected = np.flatnonzero(rows.label_present)
    x = rows.feature(variant)[selected]
    y = rows.target_normalized[selected]
    total = 0.0
    count = 0
    by_mode = {"local": [0.0, 0], "wide": [0.0, 0]}
    q_abs = []
    lower, upper = np.asarray(load_robot().limits, dtype=np.float64).T
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch):
            end = min(start + batch, len(selected))
            prediction = model(torch.from_numpy(x[start:end])).numpy()
            errors = np.sum((prediction.astype(np.float64) - y[start:end].astype(np.float64)) ** 2, axis=1)
            total += float(np.sum(errors, dtype=np.float64))
            count += len(errors)
            physical = lower + prediction.astype(np.float64) * (upper - lower)
            q_abs.extend(np.mean(np.abs(physical - rows.q_target[selected[start:end]]), axis=1).tolist())
            for mode in ("local", "wide"):
                mask = rows.mode[selected[start:end]] == mode
                by_mode[mode][0] += float(np.sum(errors[mask], dtype=np.float64))
                by_mode[mode][1] += int(mask.sum())
    if count == 0 or not math.isfinite(total):
        raise ValueError("nonfinite/empty supervised loss")
    return {"loss": total / count, "labeled": count,
            "by_mode": {mode: {"loss": (value[0] / value[1] if value[1] else None), "n": value[1]}
                        for mode, value in by_mode.items()},
            "median_joint_mae_rad": float(np.median(q_abs))}


def _save_checkpoint(path: Path, model: MLP, optimizer: torch.optim.Optimizer, variant: str,
                     seed: int, epoch: int, validation_loss: float) -> dict:
    config, schema, _ = contracts()
    inputs = read_json(INPUTS)["files"]
    by_path = {item["path"]: item["sha256"] for item in inputs}
    metadata = {
        "model_variant": variant, "input_order": config["features"][variant],
        "output_order": config["identity"]["joint_names"], "dtype": "float32",
        "architecture": "3x256_SiLU_absolute_q", "parameter_count": config["models"]["parameter_count"][variant],
        "training_seed": seed, "selected_epoch": epoch, "validation_loss": validation_loss,
        "config_sha256": sha(CONFIG), "checkpoint_schema_sha256": sha(SCHEMA),
        "c102_dataset_content_sha256": config["source"]["dataset_content_sha256"],
        "c102_manifest_sha256": by_path["experiments/C1-02/dataset-manifest.json"],
        "c102_normalization_sha256": by_path["experiments/C1-02/normalization.json"],
        "robot_urdf_sha256": by_path["assets/robots/robot_a/robot.urdf"],
        "robot_manifest_sha256": by_path["assets/robots/robot_a/manifest.json"],
        "tcp_sha256": by_path["config/robots/tcp_tool0.json"],
        "torch_fk_source_sha256": by_path["src/neurokinematics/kinematics/torch_fk.py"],
    }
    if set(schema["required_metadata"]) != set(metadata) | {"model_state_dict", "optimizer_state_dict"}:
        raise ValueError("checkpoint metadata schema mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({**metadata, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, temp)
    temp.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "accessible": True, "epoch": epoch}


def load_checkpoint(path: Path, *, expected_variant: str | None = None) -> tuple[MLP, dict]:
    if not path.is_file():
        raise ValueError("checkpoint unavailable")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config, schema, _ = contracts()
    variant = payload.get("model_variant")
    if variant not in VARIANTS or (expected_variant and variant != expected_variant):
        raise ValueError("checkpoint variant mismatch")
    if set(payload) != set(schema["required_metadata"]):
        raise ValueError("checkpoint schema mismatch")
    inputs = {item["path"]: item["sha256"] for item in read_json(INPUTS)["files"]}
    expected = {
        "input_order": config["features"][variant], "output_order": config["identity"]["joint_names"],
        "dtype": "float32", "architecture": "3x256_SiLU_absolute_q",
        "parameter_count": config["models"]["parameter_count"][variant],
        "config_sha256": sha(CONFIG), "checkpoint_schema_sha256": sha(SCHEMA),
        "c102_dataset_content_sha256": config["source"]["dataset_content_sha256"],
        "c102_manifest_sha256": inputs["experiments/C1-02/dataset-manifest.json"],
        "c102_normalization_sha256": inputs["experiments/C1-02/normalization.json"],
        "robot_urdf_sha256": inputs["assets/robots/robot_a/robot.urdf"],
        "robot_manifest_sha256": inputs["assets/robots/robot_a/manifest.json"],
        "tcp_sha256": inputs["config/robots/tcp_tool0.json"],
        "torch_fk_source_sha256": inputs["src/neurokinematics/kinematics/torch_fk.py"],
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"checkpoint metadata drift: {key}")
    model = MLP(variant)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("checkpoint nonfinite weights")
    model.eval()
    return model, {key: value for key, value in payload.items() if key not in ("model_state_dict", "optimizer_state_dict")}


def predict(model: MLP, metadata: dict, position: np.ndarray, quaternion: np.ndarray,
            q_current: np.ndarray | None) -> dict:
    config, _, norm = contracts()
    variant = metadata["model_variant"]
    if metadata["input_order"] != config["features"][variant]:
        raise ValueError("feature order mismatch")
    p = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    if p.shape != (3,) or quat.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(quat).all():
        raise ValueError("bad pose")
    if abs(float(np.linalg.norm(quat)) - 1) > 1e-12 or quat[0] < 0:
        raise ValueError("noncanonical quaternion")
    parts = [(p - np.asarray(norm["position_mean_m"])) / np.asarray(norm["position_std_m"]), quat]
    lower, upper = np.asarray(load_robot().limits, dtype=np.float64).T
    if variant == "conditioned":
        if q_current is None:
            raise ValueError("conditioned model needs q_current")
        current = np.asarray(q_current, dtype=np.float64)
        if current.shape != (6,) or not np.isfinite(current).all() or np.any(current < lower) or np.any(current > upper):
            raise ValueError("invalid q_current")
        parts.append((current - lower) / (upper - lower))
    feature = np.concatenate(parts).astype(np.float32)
    with torch.no_grad():
        output = model(torch.from_numpy(feature[None, :])).numpy()[0].astype(np.float64)
    q = lower + output * (upper - lower)
    finite = bool(np.isfinite(q).all())
    in_limits = bool(finite and np.all(q >= lower) and np.all(q <= upper))
    return {"q_raw_rad": q.tolist(), "finite": finite, "in_limits": in_limits}


def train_pair(train: Rows, validation: Rows, seed: int, *, pilot: bool, evidence: Path,
               weights: Path) -> dict:
    config, _, _ = contracts()
    validate_split_pair(train, validation)
    if seed not in config["training"]["seeds"]:
        raise ValueError("seed outside frozen protocol")
    setup(seed)
    train_idx = np.flatnonzero(train.label_present)
    val_idx = np.flatnonzero(validation.label_present)
    expected = 64 if pilot else 15204
    if len(train_idx) != expected or len(val_idx) != (32 if pilot else 3249):
        raise ValueError("labeled split budget mismatch")
    batch = config["t_c03"]["batch"] if pilot else config["training"]["effective_batch"]
    max_epoch = config["t_c03"]["epochs"] if pilot else config["training"]["max_epochs"]
    tag = "pilot" if pilot else f"seed-{seed}"
    models = {}
    optimizers = {}
    xtrain = {}
    ytrain = torch.from_numpy(train.target_normalized[train_idx])
    for variant in VARIANTS:
        torch.manual_seed(seed)
        models[variant] = MLP(variant)
        optimizers[variant] = torch.optim.AdamW(models[variant].parameters(), lr=config["optimizer"]["learning_rate"],
                                               betas=tuple(config["optimizer"]["betas"]), eps=config["optimizer"]["epsilon"],
                                               weight_decay=config["optimizer"]["weight_decay"])
        xtrain[variant] = torch.from_numpy(train.feature(variant)[train_idx])
    initial = {variant: {"train": _metrics(models[variant], train, variant),
                         "validation": _metrics(models[variant], validation, variant)} for variant in VARIANTS}
    evidence.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    log_paths = {variant: evidence / f"{tag}-{variant}-epochs.jsonl" for variant in VARIANTS}
    if any(path.exists() for path in log_paths.values()):
        raise ValueError("run log exists; preserve prior attempt")
    logs = {variant: path.open("w", encoding="utf-8", newline="\n") for variant, path in log_paths.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    best_loss = {variant: math.inf for variant in VARIANTS}
    best_epoch = {variant: 0 for variant in VARIANTS}
    best_record = {}
    stale = {variant: 0 for variant in VARIANTS}
    start = time.monotonic()
    peak_rss = _rss_bytes()
    step_count = 0
    final = {}
    try:
        for epoch in range(1, max_epoch + 1):
            permutation = rng.permutation(len(train_idx))
            for offset in range(0, len(train_idx), batch):
                selected = permutation[offset:offset + batch]
                for variant in VARIANTS:
                    model, optimizer = models[variant], optimizers[variant]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(xtrain[variant][selected])
                    loss = torch.sum((pred - ytrain[selected]) ** 2, dim=1).mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError(f"nonfinite loss {variant} epoch {epoch}")
                    loss.backward()
                    optimizer.step()
                    if any(not bool(torch.isfinite(t).all()) for t in model.parameters()):
                        raise ValueError(f"nonfinite weights {variant} epoch {epoch}")
                step_count += 1
            for variant in VARIANTS:
                model = models[variant]
                train_metrics = _metrics(model, train, variant)
                val_metrics = _metrics(model, validation, variant)
                record = {"epoch": epoch, "optimizer_steps": step_count, "train": train_metrics,
                          "validation": val_metrics, "seed": seed, "variant": variant,
                          "effective_batch": batch}
                logs[variant].write(json.dumps(record, allow_nan=False) + "\n")
                logs[variant].flush()
                final[variant] = record
                value = val_metrics["loss"]
                if not math.isfinite(value):
                    raise ValueError("nonfinite validation loss")
                if value < best_loss[variant]:
                    best_loss[variant] = value
                    best_epoch[variant] = epoch
                    stale[variant] = 0
                    best_record[variant] = _save_checkpoint(weights / f"{variant}-best.pt", model, optimizers[variant],
                                                            variant, seed, epoch, value)
                else:
                    stale[variant] += 1
            peak_rss = max(peak_rss, _rss_bytes())
            if peak_rss > config["resources"]["process_ram_cap_gib"] * (1024 ** 3):
                raise RuntimeError("C1-04 process RAM cap exceeded")
            if time.monotonic() - start > 2 * config["resources"]["per_model_seed_wall_cap_minutes"] * 60:
                raise RuntimeError("C1-04 paired wall cap exceeded")
            if not pilot and all(stale[v] >= 20 for v in VARIANTS):
                break
        last_record = {}
        for variant in VARIANTS:
            last_record[variant] = _save_checkpoint(weights / f"{variant}-last.pt", models[variant],
                                                    optimizers[variant], variant, seed, epoch,
                                                    final[variant]["validation"]["loss"])
        result = {"status": "COMPLETE", "pilot": pilot, "seed": seed,
                  "epochs": epoch, "optimizer_steps_per_model": step_count,
                  "initial": initial, "final": final, "best_epoch": best_epoch,
                  "best_validation_loss": best_loss, "best_checkpoints": best_record,
                  "last_checkpoints": last_record,
                  "epoch_logs": {v: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for v, p in log_paths.items()},
                  "elapsed_wall_s": time.monotonic() - start, "peak_rss_bytes_observed": peak_rss,
                  "train_labeled_count": len(train_idx), "validation_labeled_count": len(val_idx),
                  "train_mode_counts": dict(Counter(train.mode[train_idx].tolist())),
                  "validation_mode_counts": dict(Counter(validation.mode[val_idx].tolist()))}
        if pilot:
            gates = {}
            for variant in VARIANTS:
                a, z = initial[variant]["train"], final[variant]["train"]
                gates[variant] = {"loss_ratio": z["loss"] / a["loss"],
                                  "q_median_ratio": z["median_joint_mae_rad"] / a["median_joint_mae_rad"],
                                  "pass": z["loss"] <= 0.5 * a["loss"] and
                                          z["median_joint_mae_rad"] <= 0.75 * a["median_joint_mae_rad"]}
            result["t_c03_gates"] = gates
            result["t_c03_status"] = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
        write_json(evidence / f"{tag}-summary.json", result)
        return result
    finally:
        for handle in logs.values():
            handle.close()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p95": None, "p99": None}
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def evaluate(model: MLP, metadata: dict, rows: Rows, *, output: Path) -> dict:
    if rows.split != "validation":
        raise ValueError("C1-04 evaluation only opens validation")
    variant = metadata["model_variant"]
    inputs = load_robot()
    reference = PinocchioFK(inputs)
    independent = IndependentFK(inputs)
    lower, upper = np.asarray(inputs.limits, dtype=np.float64).T
    x = rows.feature(variant)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            predictions.append(model(torch.from_numpy(x[start:start + 1024])).numpy().astype(np.float64))
    raw = lower + np.concatenate(predictions, axis=0) * (upper - lower)
    counters = Counter()
    buckets = defaultdict(lambda: {"position_m": [], "orientation_deg": [], "q_mae_rad": [], "q_l2_rad": []})
    output.parent.mkdir(parents=True, exist_ok=True)
    spot = []
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for index, q in enumerate(raw):
            finite = bool(np.isfinite(q).all())
            in_limits = bool(finite and np.all(q >= lower) and np.all(q <= upper))
            label = bool(rows.label_present[index])
            family, mode = str(rows.family[index]), str(rows.mode[index])
            keys = ("overall", f"mode/{mode}", f"family/{family}", f"label/{label}",
                    f"family/{family}/mode/{mode}/label/{label}")
            row = {"pair_id": str(rows.pair_id[index]), "family": family, "mode": mode,
                   "label_present": label, "finite": finite, "in_limits": in_limits,
                   "q_raw_rad": q.tolist() if finite else None,
                   "position_error_m": None, "orientation_error_deg": None,
                   "q_mae_rad": None, "q_l2_rad": None}
            counters["rows"] += 1
            if not finite:
                counters["nonfinite"] += 1
            elif not in_limits:
                counters["out_of_limits"] += 1
            if finite and label:
                difference = q - rows.q_target[index]
                row["q_mae_rad"] = float(np.mean(np.abs(difference)))
                row["q_l2_rad"] = float(np.linalg.norm(difference))
                for key in keys:
                    buckets[key]["q_mae_rad"].append(row["q_mae_rad"])
                    buckets[key]["q_l2_rad"].append(row["q_l2_rad"])
            if in_limits:
                transform = reference.reference_forward_kinematics(q)
                row["position_error_m"] = float(np.linalg.norm(transform[:3, 3] - rows.position[index]))
                row["orientation_error_deg"] = float(math.degrees(rotation_error(transform[:3, :3], quaternion_rotation(rows.quaternion[index]))))
                for key in keys:
                    buckets[key]["position_m"].append(row["position_error_m"])
                    buckets[key]["orientation_deg"].append(row["orientation_error_deg"])
                counters["valid_raw"] += 1
                if len(spot) < 8:
                    cross = independent.forward_kinematics(q)
                    difference = float(np.linalg.norm(cross - transform, ord="fro"))
                    if difference > 1e-9:
                        raise ValueError("independent FK spot mismatch")
                    spot.append({"pair_id": row["pair_id"], "matrix_frobenius": difference})
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {"status": "PASS", "variant": variant, "seed": metadata["training_seed"],
               "checkpoint_epoch": metadata["selected_epoch"], "split": "validation",
               "counts": dict(counters), "rows_sha256": sha(output), "rows_bytes": output.stat().st_size,
               "rows_path": str(output), "fk_spot": spot,
               "breakdowns": {key: {metric: _stats(values) for metric, values in group.items()}
                              for key, group in sorted(buckets.items())}}
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
