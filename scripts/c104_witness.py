"""Record and reproduce fixed C1-04 validation inference without test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.neural.c104 import (CONFIG, EVIDENCE, VARIANTS, load_checkpoint,
    load_data, predict, read_json, sha, write_json)


STAGE2 = EVIDENCE / "stage2"
WITNESS = STAGE2 / "fixed-validation-inference.json"


def audit_stage1_outputs() -> int:
    lines = (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Stage1 frozen output drift: {relative}")
    return len(lines)


def sample_inputs() -> list[dict]:
    _, validation = load_data(label_fk=False)
    indices = np.flatnonzero((validation.family == "main") & (validation.mode == "local") & validation.label_present)[:10]
    if len(indices) != 10:
        raise ValueError("fixed validation witness unavailable")
    return [{"pair_id": str(validation.pair_id[i]), "position_m": validation.position[i].tolist(),
             "quaternion_wxyz": validation.quaternion[i].tolist(),
             "q_current": validation.q_current[i].tolist()} for i in indices]


def infer_all(samples: list[dict], checkpoint_paths: dict) -> dict:
    fk = PinocchioFK(load_robot())
    output = {}
    for key, item in checkpoint_paths.items():
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"checkpoint inaccessible or drifted: {key}")
        model, metadata = load_checkpoint(path, expected_variant=item["variant"])
        cases = []
        for sample in samples:
            result = predict(model, metadata, sample["position_m"], sample["quaternion_wxyz"], sample["q_current"])
            record = {"pair_id": sample["pair_id"], **result, "fk_position_m": None, "fk_rotation": None}
            if result["in_limits"]:
                transform = fk.reference_forward_kinematics(result["q_raw_rad"])
                record["fk_position_m"] = transform[:3, 3].tolist()
                record["fk_rotation"] = transform[:3, :3].tolist()
            cases.append(record)
        if sum(case["in_limits"] for case in cases) < 3:
            raise ValueError("fewer than three valid fixed validation witnesses")
        output[key] = cases
    return output


def write() -> None:
    frozen_files = audit_stage1_outputs()
    config = read_json(CONFIG)
    samples = sample_inputs()
    paths = {}
    for seed in config["training"]["seeds"]:
        run = read_json(STAGE2 / f"seed-{seed}-summary.json")
        for variant in VARIANTS:
            paths[f"{seed}/{variant}"] = {**run["best_checkpoints"][variant], "variant": variant}
    expected = infer_all(samples, paths)
    write_json(WITNESS, {"status": "RECORDED", "stage1_frozen_files": frozen_files,
                         "config_sha256": sha(CONFIG), "samples": samples,
                         "checkpoints": paths, "expected": expected,
                         "test_and_benchmark": "SEALED_NOT_RUN"})
    print(json.dumps({"status": "RECORDED", "samples": len(samples), "checkpoints": len(paths)}))


def check() -> None:
    frozen_files = audit_stage1_outputs()
    witness = read_json(WITNESS)
    if witness["stage1_frozen_files"] != frozen_files or witness["config_sha256"] != sha(CONFIG):
        raise ValueError("witness identity drift")
    actual = infer_all(witness["samples"], witness["checkpoints"])
    worst_q = worst_fk = 0.0
    for key, cases in actual.items():
        for current, expected in zip(cases, witness["expected"][key]):
            if current["pair_id"] != expected["pair_id"] or current["finite"] != expected["finite"] or current["in_limits"] != expected["in_limits"]:
                raise ValueError("witness identity/validity drift")
            if current["finite"]:
                qdiff = float(np.max(np.abs(np.asarray(current["q_raw_rad"]) - expected["q_raw_rad"])))
                worst_q = max(worst_q, qdiff)
                if qdiff > 1e-6:
                    raise ValueError("checkpoint inference drift")
            if current["in_limits"]:
                fdiff = float(np.max(np.abs(np.asarray(current["fk_rotation"]) - expected["fk_rotation"])))
                fdiff = max(fdiff, float(np.max(np.abs(np.asarray(current["fk_position_m"]) - expected["fk_position_m"]))))
                worst_fk = max(worst_fk, fdiff)
                if fdiff > 1e-9:
                    raise ValueError("independent FK witness drift")
    print(json.dumps({"status": "PASS", "frozen_files": frozen_files, "checkpoints": len(actual),
                      "samples_per_checkpoint": len(witness["samples"]),
                      "max_q_abs_rad": worst_q, "max_fk_element_abs": worst_fk,
                      "test_and_benchmark": "SEALED_NOT_RUN"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write()
    else:
        check()
