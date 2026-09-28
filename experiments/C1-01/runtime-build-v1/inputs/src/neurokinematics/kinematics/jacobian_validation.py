"""Frozen F0-03 validation and evidence CLI; not F0-04 data generation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .jacobian import IndependentJacobian
from .finite_difference import CentralDifference
from .metrics import normalized_difference, singularity_metrics
from .model import ROOT, load_robot, validate_q
from .pinocchio_jacobian import PinocchioJacobian
from .validation import environment, sample_hash, write_json

OUTPUT = ROOT / "experiments/F0-03"
CONFIG = OUTPUT / "config.json"
PAIRS = (("geometric", "pinocchio"), ("geometric", "central"), ("pinocchio", "central"))
FROZEN = {"sample_count": 256, "seed": 20260919, "bit_generator": "PCG64",
          "dtype": "float64", "primary_h_rad": 1e-6, "sensitivity_h_rad": [1e-5, 1e-7],
          "normalized_error_threshold": 1e-5, "characteristic_length_m": .9015,
          "limit_margin_rad": 2e-5, "near_threshold_fraction": .9,
          "sampling": "Generator.uniform(lower+margin, upper-margin, size=(256,6)); no replacement",
          "sample_hash_format": "SHA-256 of C-order little-endian float64 raw bytes"}


def load_config(path=CONFIG):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, expected in FROZEN.items():
        if value.get(key) != expected:
            raise ValueError(f"frozen T-F03 config mismatch: {key}")
    return value


def characteristic_length(root=ROOT):
    load_robot(root)  # authenticate manifest bytes before reading the scale
    manifest = json.loads((root / "assets/robots/robot_a/manifest.json").read_bytes())
    value = manifest["characteristic_length"]
    if value["unit"] != "m" or not np.isfinite(value["value"]) or value["value"] <= 0:
        raise ValueError("invalid manifest characteristic length")
    return float(value["value"])


def sample_configurations(inputs, config=None):
    config = load_config() if config is None else config
    bounds = np.asarray(inputs.limits, dtype=np.float64)
    margin = config["limit_margin_rad"]
    samples = np.random.Generator(np.random.PCG64(config["seed"])).uniform(
        bounds[:, 0]+margin, bounds[:, 1]-margin, (config["sample_count"], len(inputs.joint_names)))
    for q in samples:
        for sign in (-1, 1):
            validate_q(q + sign * max(config["sensitivity_h_rad"]), inputs.joint_names, inputs.limits)
    return samples


def handpicked_configurations(inputs):
    bounds = np.asarray(inputs.limits)
    result = {"zero": np.zeros(6), "midpoints": bounds.mean(axis=1),
              "lower_interior": bounds[:, 0]+2e-5, "upper_interior": bounds[:, 1]-2e-5,
              "mixed_1": np.array([.3, -.6, .8, -1., .5, -.7]),
              "mixed_2": np.array([-.4, .2, -.9, .6, -.3, 1.2]),
              "mixed_3": np.array([1.4, -2.1, 1.7, -2., 1.1, -3.2]),
              "wrist_aligned": np.array([.3, -.6, .8, -.4, 0., .7]),
              "wrist_near_aligned": np.array([.3, -.6, .8, -.4, 1e-8, .7])}
    for i, name in enumerate(inputs.joint_names):
        for sign in (-1, 1):
            q = np.zeros(6)
            q[i] = sign * .25
            result[f"{name}_{'positive' if sign > 0 else 'negative'}"] = q
    return result


def json_safe(value):
    """JSON has no infinity; preserve explicitly as 'Inf', not epsilon/null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "Inf" if value == np.inf else "-Inf" if value == -np.inf else "NaN"
    return value


def stats(values):
    finite = [x for x in values if np.isfinite(x)]
    result = dict(zip(("min", "median", "p95", "p99", "max"),
                      map(float, np.percentile(finite, [0, 50, 95, 99, 100])))) if finite else dict.fromkeys(
                          ("min", "median", "p95", "p99", "max"))
    result.update(finite_count=len(finite), infinity_count=int(sum(np.isinf(x) for x in values)),
                  percentile_scope="finite values; infinity count reported separately")
    return result


def check_jacobian(value):
    if not isinstance(value, np.ndarray) or value.shape != (6, 6) or value.dtype != np.float64:
        raise ValueError("Jacobian must be (6,6) float64")
    if not np.isfinite(value).all():
        raise ValueError("nonfinite Jacobian")


def evaluate(inputs, samples, length, h, *, methods=None, names=None):
    methods = methods or {"geometric": IndependentJacobian(inputs).jacobian,
                          "pinocchio": PinocchioJacobian(inputs).jacobian,
                          "central": CentralDifference(inputs).jacobian}
    rows, invalid, flagged = [], [], []
    for index, q in enumerate(samples):
        row = {"sample_index": index, "q": q.tolist(), "h_rad": h,
               "name": names[index] if names else str(index), "errors": {}, "metrics": {}}
        matrices = {}
        for name, method in methods.items():
            try:
                result = method(q, h) if name == "central" else method(q)
                check_jacobian(result)
                matrices[name] = result
                row["metrics"][name] = singularity_metrics(result, length)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                invalid.append({**{k: row[k] for k in ("sample_index", "q", "h_rad", "name")},
                                "method": name, "error": str(exc), "nonfinite": "nonfinite" in str(exc)})
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            row["errors"][key] = normalized_difference(matrices[a], matrices[b], length) if a in matrices and b in matrices else None
        if any(v is not None and v >= 0.9e-5 for v in row["errors"].values()):
            flagged.append(row)
        if names:
            row["jacobians"] = {k: v.tolist() for k, v in matrices.items()}
        rows.append(row)
    pairs = {}
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        valid = [r for r in rows if r["errors"][key] is not None]
        worst = max(valid, key=lambda r: r["errors"][key]) if valid else None
        pairs[key] = {"N": len(rows), "valid_count": len(valid),
                      "normalized_error": stats([r["errors"][key] for r in valid]),
                      "threshold_exceeded_count": sum(r["errors"][key] > 1e-5 for r in valid),
                      "invalid_result_count": len(rows)-len(valid),
                      "nonfinite_result_count": len({e["sample_index"] for e in invalid if e["method"] in (a, b) and e["nonfinite"]}),
                      "worst": None if worst is None else {k: worst[k] for k in ("sample_index", "name", "q", "errors", "metrics")}}
    metric_stats = {name: {key: stats([r["metrics"][name][key] for r in rows if name in r["metrics"]])
                          for key in ("sigma_min", "sigma_max", "condition", "manipulability")}
                    for name in methods}
    ok = not invalid and all(p["threshold_exceeded_count"] == 0 for p in pairs.values())
    return {"h_rad": h, "status": "PASS" if ok else "FAIL", "pairs": pairs,
            "singularity_summary": metric_stats}, {"invalid_results": invalid, "failed_or_near_threshold": flagged}, rows


def run_validation(inputs=None, *, methods=None):
    inputs = load_robot() if inputs is None else inputs
    config = load_config()
    length = characteristic_length()
    if config["characteristic_length_m"] != length:
        raise ValueError("config/manifest scale mismatch")
    samples = sample_configurations(inputs, config)
    selected = handpicked_configurations(inputs)
    summary = {"schema_version": 1, "task": "F0-03", "test_id": "T-F03",
               "sample_count": len(samples), "sample_shape": list(samples.shape),
               "seed": config["seed"], "dtype": str(samples.dtype), "bit_generator": "PCG64",
               "sample_sha256": sample_hash(samples), "sample_hash_format": config["sample_hash_format"],
               "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
               "input_hashes": inputs.hashes, "characteristic_length_m": length,
               "normalized_error_threshold": 1e-5, "primary_h_rad": 1e-6, "runs": {}}
    diagnostics, handpicked = {}, {}
    for h in (1e-5, 1e-6, 1e-7):
        key = format(h, ".0e")
        report, diag, rows = evaluate(inputs, samples, length, h, methods=methods)
        hand, hand_diag, hand_rows = evaluate(inputs, np.array(list(selected.values())), length, h,
                                              methods=methods, names=list(selected))
        low = min((r for r in rows if "geometric" in r["metrics"]),
                  key=lambda r: r["metrics"]["geometric"]["sigma_min"], default=None)
        report["handpicked"] = hand
        report["low_sigma_sample"] = low
        summary["runs"][key] = report
        diagnostics[key] = {"random": diag, "handpicked": hand_diag,
                            "low_sigma_sample": low,
                            "singular_subgroup": [r for r in hand_rows if r["name"] in ("zero", "wrist_aligned", "wrist_near_aligned")]}
        handpicked[key] = hand_rows
    summary["status"] = "PASS" if all(r["status"] == "PASS" and r["handpicked"]["status"] == "PASS"
                                                 for r in summary["runs"].values()) else "FAIL"
    return json_safe(summary), json_safe(diagnostics), json_safe(handpicked)


def inspect(inputs):
    a, b = IndependentJacobian(inputs), PinocchioJacobian(inputs)
    return {"shape": [6, 6], "dtype": "float64", "row_order": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "point": inputs.tcp, "axes": inputs.base, "tip": inputs.tip,
            "units": {"linear": "m/rad", "angular": "rad/rad"},
            "pinocchio_reference_frame": "LOCAL_WORLD_ALIGNED",
            "base_conversion": "R_world_base.T applied separately to Motion.linear and Motion.angular; no point shift",
            "reference_documentation": "https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/group__pinocchio__multibody.html",
            "so3_increment": "log(R_plus @ R_minus.T)/(2h); vee=(R32-R23,R13-R31,R21-R12)/2; base axes",
            "pi_branch": "principal angle; exact pi sign ambiguous, largest axis component positive",
            "joint_names": list(inputs.joint_names), "q_indices": b.q_indices, "v_indices": b.v_indices,
            "frame_ids": b.frame_ids, "characteristic_length_m": characteristic_length(),
            "chain": [{"name": j.name, "kind": j.kind, "parent": j.parent, "child": j.child}
                      for j in a.fk.chain], "input_hashes": inputs.hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "validate"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        inputs = load_robot()
        contract = inspect(inputs)
        write_json(args.output / "chain-and-frame-contract.json", contract)
        if args.command == "inspect":
            print(json.dumps(contract, indent=2))
            return 0
        summary, diagnostics, selected = run_validation(inputs)
        write_json(args.output / "diagnostics.json", diagnostics)
        write_json(args.output / "handpicked-jacobians.json", selected)
        write_json(args.output / "sample-hash.json", {k: summary[k] for k in
                   ("sample_sha256", "sample_shape", "seed", "dtype", "bit_generator", "sample_hash_format")})
        write_json(args.output / "jacobian-sensitivity.json", {"sample_sha256": summary["sample_sha256"],
                   "config_sha256": summary["config_sha256"], "input_hashes": summary["input_hashes"],
                   "runs": summary["runs"]})
        write_json(args.output / "environment.json", environment())
        write_json(args.output / "jacobian-validation-summary.json", summary)
        print(json.dumps({"status": summary["status"], "sample_sha256": summary["sample_sha256"],
              "max_errors": {h: {k: p["normalized_error"]["max"] for k, p in r["pairs"].items()}
                             for h, r in summary["runs"].items()}}, indent=2))
        return int(summary["status"] != "PASS")
    except (ValueError, RuntimeError, OSError) as exc:
        write_json(args.output / "jacobian-validation-summary.json", {"status": "INCONCLUSIVE", "error": str(exc)})
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
