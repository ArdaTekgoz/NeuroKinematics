"""Immutable validation/freeze/new-final/audit commands for ADR-026."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np

from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error, singularity_metrics
from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.solvers.analytic import AnalyticIK

BASE = ROOT / "experiments/C1-06S"
RAW = ROOT / "data/generated/C1-06S"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def frozen_integrity():
    mapping = read(ROOT / "experiments/C1-06R/training-freeze.json")["files"]
    for name, digest in mapping.items():
        if sha(ROOT / name) != digest:
            raise ValueError("historical source/input drift: " + name)
    return len(mapping)


def metrics(q, request, fk, limits):
    if q is None:
        return dict(valid=False, profile_a=False, profile_b=False, position_m=None, orientation_deg=None)
    q = np.asarray(q, dtype=np.float64)
    valid = q.shape == (6,) and np.isfinite(q).all() and (q >= limits[:, 0]).all() and (q <= limits[:, 1]).all()
    if not valid:
        return dict(valid=False, profile_a=False, profile_b=False, position_m=None, orientation_deg=None)
    pose = fk(q)
    p = float(np.linalg.norm(pose[:3, 3] - request["position_m"]))
    angle = math.degrees(rotation_error(pose[:3, :3], quaternion_rotation(request["quaternion_wxyz"])))
    return dict(valid=True, profile_a=p <= .002 and angle <= 1., profile_b=p <= .001 and angle <= .5,
                position_m=p, orientation_deg=angle)


def quantiles(values):
    values = sorted(float("inf") if value is None else value for value in values)
    return {name: (values[math.ceil(p * len(values)) - 1] if math.isfinite(values[math.ceil(p * len(values)) - 1]) else None)
            for name, p in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.))}


def summary(records):
    first = [r for r in records if r["pass"] == 0]
    result = {"unique_queries": len(first), "measurements": len(records), "groups": {}}
    keys = ["all", "main", "boundary", "singularity", "local", "wide", "missing_teacher"]
    keys += [f"{family}/{mode}" for family in ("main", "boundary", "singularity") for mode in ("local", "wide")]
    for key in keys:
        rows = [r for r in first if key == "all" or r["family"] == key or r["mode"] == key
                or f'{r["family"]}/{r["mode"]}' == key or (key == "missing_teacher" and not r.get("label_present", True))]
        if not rows:
            continue
        ids = {r["id"] for r in rows}
        measured = [r for r in records if r["id"] in ids]
        n, a = len(rows), sum(r["profile_a"] for r in rows)
        z = 1.959963984540054
        low = ((a/n + z*z/(2*n)) - z * math.sqrt(a/n*(1-a/n)/n + z*z/(4*n*n))) / (1+z*z/n)
        result["groups"][key] = dict(n=n, profile_a=a, profile_b=sum(r["profile_b"] for r in rows),
            profile_a_rate=a/n, profile_a_wilson95_lower=low,
            statuses=dict(Counter(r["status"] for r in rows)), invalid=sum(not r["valid"] for r in rows),
            position_m=quantiles([r["position_m"] for r in rows]),
            orientation_deg=quantiles([r["orientation_deg"] for r in rows]),
            component_total_ms=quantiles([r["elapsed_ns"]/1e6 for r in measured]),
            measured_within_10ms=sum(r["profile_a"] and r["elapsed_ns"] <= 10_000_000 for r in measured),
            measured_within_50ms=sum(r["profile_a"] and r["elapsed_ns"] <= 50_000_000 for r in measured))
    return result


def evaluate(requests, directory, passes):
    directory.mkdir(parents=True, exist_ok=False)
    solver, reference = AnalyticIK(), PinocchioFK(load_robot())
    limits = np.asarray(solver.inputs.limits)
    records, predictions = [], {}
    for request in requests[:20]:
        solver.solve(request["q_current_rad"], request["position_m"], request["quaternion_wxyz"])
    with (directory / "results.jsonl").open("x", encoding="utf-8") as stream:
        for repeat in range(passes):
            for request in requests:
                # Only the actual public request crosses the solver boundary.
                start = perf_counter_ns()
                solved = solver.solve(request["q_current_rad"], request["position_m"], request["quaternion_wxyz"])
                elapsed = perf_counter_ns() - start
                m = metrics(solved.q_candidate, request, reference.reference_forward_kinematics, limits)
                if (solved.status == "KINEMATIC_VALID") != m["profile_b"]:
                    raise ValueError("independent false acceptance/status mismatch")
                q = None if solved.q_candidate is None else solved.q_candidate.tolist()
                if repeat and predictions[request["id"]] != q:
                    raise ValueError("non-deterministic candidate across passes")
                predictions[request["id"]] = q
                row = dict(id=request["id"], family=request["family"], mode=request["mode"],
                           label_present=request.get("label_present", True), **m, q_rad=q,
                           status=solved.status, elapsed_ns=elapsed, component_elapsed_ns=solved.elapsed_ns,
                           candidates=solved.generated_candidates, checked=solved.checked_candidates,
                           **{"pass": repeat})
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                records.append(row)
            print(f"pass {repeat + 1}: A {sum(r['profile_a'] for r in records if r['pass']==repeat)}/{len(requests)}", flush=True)
    return records


def validation():
    from neurokinematics.neural.c104 import load_data
    frozen_count = frozen_integrity()
    _, rows = load_data(label_fk=True)
    requests = [dict(id=str(rows.pair_id[i]), family=str(rows.family[i]), mode=str(rows.mode[i]),
        q_current_rad=rows.q_current[i].tolist(), position_m=rows.position[i].tolist(),
        quaternion_wxyz=rows.quaternion[i].tolist(), label_present=bool(rows.label_present[i])) for i in range(len(rows.pair_id))]
    records = evaluate(requests, RAW / "validation", 1)
    write(RAW / "validation/requests.json", requests)
    write(BASE / "validation.json", dict(status="COMPLETE", frozen_files=frozen_count, **summary(records),
        raw_sha256=sha(RAW/"validation/results.jsonl"), requests_sha256=sha(RAW/"validation/requests.json"),
        solver_sha256=sha(ROOT/"src/neurokinematics/solvers/analytic.py"), old_final="NOT_READ"))


def freeze():
    frozen_count = frozen_integrity()
    val = read(BASE / "validation.json")
    if val["groups"]["main"]["profile_a_rate"] < .95:
        raise ValueError("validation gate failed")
    if val["solver_sha256"] != sha(ROOT/"src/neurokinematics/solvers/analytic.py"):
        raise ValueError("solver changed since validation")
    unit = BASE / "unit-r3.xml"
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(unit).getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")):
        raise ValueError("unit gate failed")
    files = [Path(__file__), ROOT/"src/neurokinematics/solvers/analytic.py", BASE/"protocol.json", unit,
             ROOT/"tests/c1_06s/test_analytic.py", BASE/"validation.json",
             ROOT/"docs/adr/ADR-026-geometric-core-recovery.md"]
    files += list((ROOT/"src/neurokinematics/kinematics").glob("*.py"))
    files += [ROOT/"pixi.lock", ROOT/"experiments/F0-05/config.json"]
    write(BASE/"final-freeze.json", dict(status="FROZEN_BEFORE_FINAL_GENERATION", utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        files={str(p.relative_to(ROOT)).replace("\\", "/"):sha(p) for p in files}, frozen_files=frozen_count,
        python=sys.version, numpy=np.__version__, platform=platform.platform(), old_final="NOT_READ"))


def verify_freeze():
    for name, digest in read(BASE/"final-freeze.json")["files"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("new final source drift: " + name)
    return frozen_integrity()


def generate_final(*, persist=True):
    cfg, protocol = read(ROOT/"experiments/F0-05/config.json"), read(BASE/"protocol.json")
    robot = load_robot()
    bounds = np.asarray(robot.limits)
    reference, independent, jac = PinocchioFK(robot), IndependentFK(robot), IndependentJacobian(robot)
    rngs = [np.random.default_rng(protocol["final_seed"] + i) for i in range(5)]
    # Existing development inputs are excluded without opening historical final data.
    from neurokinematics.neural.c104 import load_data
    train, validation_rows = load_data(label_fk=False)
    excluded = {np.asarray(q, dtype="<f8").tobytes() for rows in (train, validation_rows)
                for q in rows.q_target[rows.label_present]}
    used, requests, witnesses, counts = set(), [], [], Counter()
    for family_idx, (family, n) in enumerate(protocol["query_counts"].items()):
        target_rng = rngs[family_idx]
        for index in range(n):
            for attempt in range(cfg["target_candidate_limits"][family]):
                q = target_rng.uniform(bounds[:, 0], bounds[:, 1])
                if family == "boundary":
                    j, side = (index//2) % 6, index % 2
                    d = target_rng.random() * cfg["boundary_normalized_limit_distance_lt"] * (bounds[j, 1]-bounds[j, 0])
                    q[j] = bounds[j, side] + (d if side == 0 else -d)
                if family == "singularity":
                    sigma = singularity_metrics(jac.jacobian(q), cfg["characteristic_length_m"])["sigma_min"]
                    if sigma > cfg["singularity_sigma_min_le"]:
                        continue
                key = np.asarray(q, dtype="<f8").tobytes()
                if key not in excluded and key not in used:
                    break
            else:
                raise ValueError("target generation cap")
            counts[family+"_target_candidates"] += attempt + 1
            mode = "local" if index % 2 == 0 else "wide"
            for start_attempt in range(cfg["start_candidate_limit_per_query"]):
                if mode == "local":
                    current = q + rngs[3].uniform(*cfg["local_perturbation_uniform_rad"], size=6)
                    if (current < bounds[:, 0]).any() or (current > bounds[:, 1]).any() or np.array_equal(q, current):
                        continue
                else:
                    current = rngs[4].uniform(bounds[:, 0], bounds[:, 1])
                    if np.sqrt(np.mean(((current-q)/(bounds[:, 1]-bounds[:, 0]))**2)) < cfg["wide_min_normalized_rms_distance"]:
                        continue
                break
            else:
                raise ValueError("start generation cap")
            target = reference.reference_forward_kinematics(q)
            check = independent.forward_kinematics(q)
            if np.linalg.norm(target[:3, 3]-check[:3, 3]) > 1e-9 or np.linalg.norm(target[:3, :3]-check[:3, :3]) > 1e-9:
                raise ValueError("target FK disagreement")
            qid = f"c106s-final1-{family}-{index:05d}"
            requests.append(dict(id=qid, family=family, mode=mode, q_current_rad=current.tolist(),
                position_m=target[:3, 3].tolist(), quaternion_wxyz=canonical_quaternion(target[:3, :3]).tolist()))
            witnesses.append(dict(id=qid, q_target=q.tolist(), target_attempts=attempt+1, start_attempts=start_attempt+1))
            used.add(key)
        print("generated", family, n, flush=True)
    if persist:
        write(RAW/"final1-inputs/requests.json", requests)
        write(RAW/"final1-inputs/witnesses.json", witnesses)
        write(BASE/"final1-inputs.json", dict(status="GENERATED_BEFORE_EVALUATION", unique_targets=len(used),
            development_exact_duplicates=0, old_final="NOT_READ", counts=dict(counts),
            requests_sha256=sha(RAW/"final1-inputs/requests.json"), witnesses_sha256=sha(RAW/"final1-inputs/witnesses.json")))
    else:
        if requests != read(RAW/"final1-inputs/requests.json") or witnesses != read(RAW/"final1-inputs/witnesses.json"):
            raise ValueError("query/witness generation replay mismatch")
    return requests


def final():
    verify_freeze()
    requests = generate_final()
    records = evaluate(requests, RAW/"final1", read(BASE/"protocol.json")["passes"])
    verify_freeze()
    result = summary(records)
    passed = result["groups"]["main"]["profile_a_rate"] >= read(BASE/"protocol.json")["main_profile_a_min_rate"]
    write(BASE/"final1.json", dict(status="PASS" if passed else "FAIL", **result,
        raw_sha256=sha(RAW/"final1/results.jsonl"), false_acceptance=0, freeze_sha256=sha(BASE/"final-freeze.json"),
        learned=False, old_h2="REJECTED_UNCHANGED", h1="NOT_MEASURED", collision="NOT_CHECKED"))


def audit(mode):
    if mode == "final1":
        verify_freeze()
    requests_path = RAW/("final1-inputs" if mode == "final1" else mode)/"requests.json"
    requests = {r["id"]:r for r in read(requests_path)}
    if len(requests) != len(read(requests_path)):
        raise ValueError("duplicate query IDs")
    input_record = read(BASE/("final1-inputs.json" if mode == "final1" else "validation.json"))
    if input_record["requests_sha256"] != sha(requests_path):
        raise ValueError("request SHA disagreement")
    if mode == "final1":
        if input_record["witnesses_sha256"] != sha(RAW/"final1-inputs/witnesses.json"):
            raise ValueError("witness SHA disagreement")
        generate_final(persist=False)
    reference = IndependentFK(load_robot())
    limits = np.asarray(load_robot().limits)
    records = [json.loads(line) for line in (RAW/mode/"results.jsonl").read_text().splitlines()]
    expected_passes = read(BASE/"protocol.json")["passes"] if mode == "final1" else 1
    seen, maxima = set(), dict(position_m=0., orientation_deg=0.)
    for record in records:
        key = record["id"], record["pass"]
        if key in seen or record["pass"] not in range(expected_passes):
            raise ValueError("duplicate or unexpected measurement")
        seen.add(key)
        request = requests[record["id"]]
        if record["family"] != request["family"] or record["mode"] != request["mode"]:
            raise ValueError("stratum disagreement")
        measured = metrics(record["q_rad"], request, reference.forward_kinematics, limits)
        if (record["status"] == "KINEMATIC_VALID") != measured["profile_b"]:
            raise ValueError("status disagreement")
        for field in ("valid", "profile_a", "profile_b"):
            if record[field] != measured[field]:
                raise ValueError("independent classification disagreement")
        for field, tolerance in (("position_m", 1e-9), ("orientation_deg", 5e-6)):
            if measured[field] is not None:
                delta = abs(record[field]-measured[field])
                maxima[field] = max(maxima[field], delta)
                if delta > tolerance:
                    raise ValueError("independent metric disagreement")
    if len(seen) != len(requests)*expected_passes:
        raise ValueError("incomplete denominator")
    stored = read(BASE/(mode+".json"))
    if stored["raw_sha256"] != sha(RAW/mode/"results.jsonl") or stored["groups"] != summary(records)["groups"]:
        raise ValueError("summary/hash disagreement")
    write(BASE/(mode+"-audit.json"), dict(status="PASS", measurements=len(records), unique_queries=len(requests),
        maximum_metric_difference=maxima, raw_sha256=sha(RAW/mode/"results.jsonl"),
        requests_sha256=sha(requests_path), frozen_files=frozen_integrity(), old_final="NOT_READ",
        input_replay="EXACT" if mode == "final1" else "HISTORICAL_VALIDATION"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validation", "freeze", "final", "audit-validation", "audit-final1"])
    action = parser.parse_args().action
    if action.startswith("audit-"):
        audit(action.removeprefix("audit-"))
    else:
        globals()[action]()
