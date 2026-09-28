"""Frozen T-F02 validation, diagnostics and evidence CLI (not a dataset factory)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pinocchio as pin

from .custom_fk import IndependentFK
from .model import ROOT, SOURCE_COMMIT, load_robot, validate_q
from .pinocchio_fk import PinocchioFK
from .transforms import check_transform

CONFIG_PATH = ROOT / "experiments/F0-02/config.json"
SEED = 20260918
POSITION_THRESHOLD = 1e-9
ROTATION_THRESHOLD = 1e-9


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"sample_count": 10000, "seed": SEED, "bit_generator": "PCG64",
                "dtype": "float64", "position_threshold_m": POSITION_THRESHOLD,
                "rotation_frobenius_threshold": ROTATION_THRESHOLD,
                "near_threshold_fraction": 0.9,
                "sampling": "Generator.uniform(lower_rad, upper_rad, size=(N,6))"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen T-F02 config mismatch: {key}")
    return config


def sample_configurations(inputs, count=10000, seed=SEED):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("sample count must be positive")
    bounds = np.asarray(inputs.limits, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = generator.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(inputs.joint_names)))
    for q in samples:
        validate_q(q, inputs.joint_names, inputs.limits)
    return samples


def sample_hash(samples):
    return hashlib.sha256(np.asarray(samples, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def compare(custom, reference):
    check_transform(custom)
    check_transform(reference)
    return {
        "position_error_m": float(np.linalg.norm(custom[:3, 3] - reference[:3, 3], ord=2)),
        "rotation_frobenius_error": float(np.linalg.norm(custom[:3, :3] - reference[:3, :3], ord="fro")),
        "max_absolute_matrix_error": float(np.max(np.abs(custom - reference))),
    }


def handpicked_configurations(inputs):
    bounds = np.asarray(inputs.limits)
    result = {"zero": np.zeros(6), "midpoints": bounds.mean(axis=1),
              "lower_near": bounds[:, 0] + 1e-8, "upper_near": bounds[:, 1] - 1e-8,
              "lower_exact": bounds[:, 0], "upper_exact": bounds[:, 1],
              "mixed_1": np.array([0.3, -0.6, 0.8, -1.0, 0.5, -0.7]),
              "mixed_2": np.array([-0.4, 0.2, -0.9, 0.6, -0.3, 1.2]),
              "mixed_3": np.array([1.4, -2.1, 1.7, -2.0, 1.1, -3.2])}
    for index, name in enumerate(inputs.joint_names):
        for sign in (-1, 1):
            q = np.zeros(6)
            q[index] = sign * 0.25
            result[f"{name}_{'positive' if sign > 0 else 'negative'}"] = q
    return result


def _stats(values):
    if not values:
        return dict.fromkeys(("min", "median", "p95", "p99", "max"))
    return dict(zip(("min", "median", "p95", "p99", "max"),
                    map(float, np.percentile(values, [0, 50, 95, 99, 100]))))


def run_validation(inputs, count=10000, *, custom=None, reference=None):
    """Keep every failure; no resampling. Nonfinite/invalid outputs cannot pass."""
    load_config()
    custom = IndependentFK(inputs) if custom is None else custom
    reference = PinocchioFK(inputs) if reference is None else reference
    samples = sample_configurations(inputs, count)
    errors, flagged, invalid = [], [], []
    worst_position, worst_rotation = None, None
    nonfinite_count = 0
    for index, q in enumerate(samples):
        record = {"sample_index": index, "q": q.tolist()}
        try:
            a = custom.forward_kinematics(q)
            b = reference.reference_forward_kinematics(q)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                nonfinite_count += 1
            metrics = compare(a, b)
        except (ValueError, RuntimeError) as exc:
            record["error"] = str(exc)
            invalid.append(record)
            continue
        record.update(metrics)
        record.update(custom_translation=a[:3, 3].tolist(), reference_translation=b[:3, 3].tolist())
        errors.append((metrics["position_error_m"], metrics["rotation_frobenius_error"]))
        if worst_position is None or record["position_error_m"] > worst_position["position_error_m"]:
            worst_position = record
        if worst_rotation is None or record["rotation_frobenius_error"] > worst_rotation["rotation_frobenius_error"]:
            worst_rotation = record
        if record["position_error_m"] >= 0.9 * POSITION_THRESHOLD or record["rotation_frobenius_error"] >= 0.9 * ROTATION_THRESHOLD:
            flagged.append(record)
    exceeded = sum(p > POSITION_THRESHOLD or r > ROTATION_THRESHOLD for p, r in errors)
    summary = {
        "schema_version": 1, "task": "F0-02", "requirement": "REQ-F02", "test_id": "T-F02",
        "status": "FAIL" if invalid or exceeded else ("PASS" if count == 10000 else "INCONCLUSIVE"),
        "scope": "final acceptance" if count == 10000 else "smoke only; cannot close T-F02",
        "robot_id": inputs.robot_id, "input_hashes": inputs.hashes, "source_commit": SOURCE_COMMIT,
        "frames": {"base": inputs.base, "tip": inputs.tip, "tcp": inputs.tcp},
        "joint_names": list(inputs.joint_names), "limits_rad": [list(x) for x in inputs.limits],
        "sample_count": count, "sample_shape": list(samples.shape), "seed": SEED,
        "bit_generator": "PCG64", "sample_sha256": sample_hash(samples), "dtype": str(samples.dtype),
        "sample_hash_format": load_config()["sample_hash_format"],
        "thresholds": {"position_m": POSITION_THRESHOLD, "rotation_frobenius": ROTATION_THRESHOLD},
        "position_error_m": _stats([p for p, _ in errors]),
        "rotation_frobenius_error": _stats([r for _, r in errors]),
        "threshold_exceeded_count": exceeded, "invalid_result_count": len(invalid),
        "nonfinite_result_count": nonfinite_count, "valid_comparison_count": len(errors),
        "worst_position": worst_position, "worst_rotation": worst_rotation,
    }
    diagnostics = {"failed_or_near_threshold": flagged, "invalid_results": invalid,
                   "near_threshold_fraction": 0.9}
    return summary, diagnostics


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def inspect(inputs):
    independent, reference = IndependentFK(inputs), PinocchioFK(inputs)
    return {"input_hashes": inputs.hashes,
            "chain": [{"name": j.name, "kind": j.kind, "parent": j.parent, "child": j.child}
                      for j in independent.chain],
            "joint_names": list(inputs.joint_names), "pinocchio_q_indices": reference.q_indices,
            "pinocchio_body_frame_ids": reference.frame_ids}


def environment():
    return {"python": platform.python_version(), "os": platform.platform(),
            "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "ram": "NOT_MEASURED", "gpu": "NOT_USED", "execution": "serial CPU loop",
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pytest", "xacro")},
            "pinocchio": str(pin.__version__),
            "pixi": subprocess.check_output(["pixi", "--version"], text=True).strip(),
            "lock_sha256": hashlib.sha256((ROOT / "pixi.lock").read_bytes()).hexdigest(),
            "thread_environment": {k: os.environ.get(k, "UNSET") for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "linux_execution": "NOT_RUN"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "validate"])
    parser.add_argument("--small", action="store_true", help="1000-sample smoke; never final PASS")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/F0-02")
    args = parser.parse_args(argv)
    try:
        inputs = load_robot()
        if args.command == "inspect":
            print(json.dumps(inspect(inputs), indent=2))
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        summary, diagnostics = run_validation(inputs, 1000 if args.small else 10000)
        prefix = "smoke-" if args.small else ""
        write_json(args.output / f"{prefix}diagnostics.json", diagnostics)
        write_json(args.output / f"{prefix}sample-hash.json",
                   {k: summary[k] for k in ("sample_sha256", "sample_shape", "sample_hash_format", "seed", "bit_generator", "dtype")})
        a, b = IndependentFK(inputs), PinocchioFK(inputs)
        selected = []
        for name, q in handpicked_configurations(inputs).items():
            ta, tb = a.forward_kinematics(q), b.reference_forward_kinematics(q)
            selected.append({"name": name, "q": q.tolist(), "custom_transform": ta.tolist(),
                             "reference_transform": tb.tolist(), **compare(ta, tb)})
        write_json(args.output / f"{prefix}handpicked-results.json", selected)
        write_json(args.output / f"{prefix}chain-inspection.json", inspect(inputs))
        write_json(args.output / f"{prefix}environment.json", environment())
        write_json(args.output / f"{prefix}execution.json", {"started_utc": started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        # Publish the decision last, after the complete supporting package exists.
        write_json(args.output / f"{prefix}fk-validation-summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 1 if summary["status"] == "FAIL" else 0
    except (ValueError, OSError, RuntimeError) as exc:
        failure = {"status": "INCONCLUSIVE", "error": str(exc)}
        if args.command == "validate" and args.output.is_dir():
            prefix = "smoke-" if args.small else ""
            write_json(args.output / f"{prefix}fk-validation-summary.json", failure)
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
