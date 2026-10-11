"""Replay a frozen geometric result from a separate source checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error
from neurokinematics.solvers.analytic import AnalyticIK


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() == args.artifacts.resolve():
        raise ValueError("separate source checkout required")
    manifest = json.loads((args.artifacts/"experiments/C1-06S/final-freeze.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("frozen source mismatch: "+name)
    requests_path = args.artifacts/"data/generated/C1-06S/final1-inputs/requests.json"
    results_path = args.artifacts/"data/generated/C1-06S/final1/results.jsonl"
    expected_inputs = json.loads((args.artifacts/"experiments/C1-06S/final1-inputs.json").read_text())
    expected_results = json.loads((args.artifacts/"experiments/C1-06S/final1.json").read_text())
    if sha(requests_path) != expected_inputs["requests_sha256"] or sha(results_path) != expected_results["raw_sha256"]:
        raise ValueError("artifact SHA mismatch")
    requests = json.loads(requests_path.read_text())
    expected = {}
    for line in results_path.read_text().splitlines():
        row = json.loads(line)
        if row["pass"] == 0:
            expected[row["id"]] = row
    solver, reference = AnalyticIK(), PinocchioFK(load_robot())
    checked = 0
    for request in requests:
        result = solver.solve(request["q_current_rad"], request["position_m"], request["quaternion_wxyz"])
        old = expected[request["id"]]
        q = None if result.q_candidate is None else result.q_candidate.tolist()
        if q != old["q_rad"] or result.status != old["status"]:
            raise ValueError("candidate/status replay mismatch")
        if q is not None:
            pose = reference.reference_forward_kinematics(q)
            p = float(np.linalg.norm(pose[:3, 3]-request["position_m"]))
            angle = float(np.degrees(rotation_error(pose[:3, :3], quaternion_rotation(request["quaternion_wxyz"]))))
            if p != old["position_m"] or angle != old["orientation_deg"]:
                raise ValueError("reference FK metric replay mismatch")
        checked += 1
    record = dict(status="PASS", unique_queries=checked, source_root=str(ROOT),
                  executable=sys.executable, source_files=len(manifest["files"]),
                  candidate_replay="EXACT", metric_replay="EXACT", runtime="EXISTING_LOCKED_RUNTIME",
                  fresh_environment="NOT_CREATED", requests_sha256=sha(requests_path), raw_sha256=sha(results_path))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
