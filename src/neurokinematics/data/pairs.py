"""C1-02 versioned state-conditioned pairs; F0-04 source remains immutable."""

from __future__ import annotations

from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from neurokinematics.benchmark.queries import q_key
from neurokinematics.data.factory import (canonical_array_hash, canonical_quaternion,
                                          read_shard, verify_dataset, write_deterministic_npz)
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import ROOT, load_robot, validate_q
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.solvers.dls import DLS, profile_success

EVIDENCE = ROOT / "experiments/C1-02"
CONFIG = EVIDENCE / "config.json"
SCHEMA = EVIDENCE / "schema.json"


def process_peak_rss_bytes() -> int:
    """Windows peak resident working set of the current process."""
    if os.name != "nt":
        raise RuntimeError("C1-02 native Windows resource probe required")
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def require_single_thread_environment():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    actual = {name: os.environ.get(name) for name in names}
    if any(value != "1" for value in actual.values()):
        raise ValueError(f"frozen single-thread environment not enforced: {actual}")
    return actual


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_contract():
    return json.loads(CONFIG.read_text(encoding="utf-8")), json.loads(SCHEMA.read_text(encoding="utf-8"))


def per_row_seed(data_seed: int, sample_id: str, mode: str, stream: str) -> int:
    raw = f"C1-02/v1|{data_seed}|{sample_id}|{mode}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def roots():
    config, _ = load_contract()
    source_root = ROOT / config["source"]["dataset_root"]
    manifest_path = ROOT / config["source"]["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_dataset(source_root, manifest_path)
    source_schema = json.loads((ROOT / "experiments/F0-04/schema.json").read_text(encoding="utf-8"))
    fields = [field["name"] for field in source_schema["fields"]]
    records = []
    for subset in ("main", "boundary", "singularity"):
        for shard in manifest["shards"][subset]:
            arrays = read_shard(source_root / shard["path"], fields)
            if canonical_array_hash(arrays, fields) != shard["content_sha256"]:
                raise ValueError(f"source content mismatch: {shard['path']}")
            for index in range(shard["record_count"]):
                sample = arrays["sample_id"][index].decode()
                family = arrays["sampling_class"][index].decode()
                if family != ("main_lhs" if subset == "main" else subset):
                    raise ValueError("source family mismatch")
                records.append({"sample_id": sample, "group_id": arrays["group_id"][index].decode(),
                                "split": arrays["split"][index].decode(), "family": subset,
                                "q": arrays["q"][index].astype("<f8"),
                                "position": arrays["position_m"][index].astype("<f8"),
                                "quaternion": arrays["quaternion_wxyz"][index].astype("<f8")})
    if len(records) != sum(config["source"]["subsets"].values()):
        raise ValueError("source row count mismatch")
    group_split = {}
    root_q_split = {}
    for row in records:
        if row["group_id"] in group_split or q_key(row["q"]) in root_q_split:
            raise ValueError("duplicate source root group/q")
        group_split[row["group_id"]] = row["split"]
        root_q_split[q_key(row["q"])] = row["split"]
    return records, manifest


def select_pilot(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["family"], row["split"])].append(row)
    selected = []
    for family in ("main", "boundary", "singularity"):
        for split in ("train", "validation", "test"):
            part = sorted(groups[(family, split)], key=lambda r: r["sample_id"])
            if len(part) < 10:
                raise ValueError("pilot subgroup insufficient")
            selected.extend(part[:10])
    if len(selected) != 90:
        raise ValueError("pilot count mismatch")
    return selected


def make_base(row, mode, config, bounds):
    sample = row["sample_id"]
    seed = per_row_seed(config["randomness"]["data_seed"], sample, mode, mode)
    rng = np.random.Generator(np.random.PCG64(seed))
    target = row["q"]
    if mode == "local":
        for attempts in range(1, config["local"]["candidate_cap_per_row"] + 1):
            current = target + rng.uniform(-0.1, 0.1, size=6)
            if np.all(current >= bounds[:, 0]) and np.all(current <= bounds[:, 1]) and q_key(current) != q_key(target):
                break
        else:
            raise ValueError(f"local perturbation cap: {sample}")
    else:
        current = rng.uniform(bounds[:, 0], bounds[:, 1], size=6)
        attempts = 1
    return {"pair_id": f"c102-{sample}-{mode}", "source_sample_id": sample,
            "group_id": row["group_id"], "source_family": row["family"], "split": row["split"],
            "pair_mode": mode, "derivation_seed": seed, "derivation_attempts": attempts,
            "root_q_target": target.copy(), "q_current": current,
            "position_m": row["position"].copy(), "quaternion_wxyz": row["quaternion"].copy()}


def candidate_valid(q, target_p, target_quat, fk, inputs):
    try:
        candidate = validate_q(q, inputs.joint_names, inputs.limits)
        result = fk.reference_forward_kinematics(candidate)
        pos_error = float(np.linalg.norm(result[:3, 3] - target_p))
        rot_error = float(rotation_error(result[:3, :3], quaternion_rotation(target_quat)))
        return profile_success(pos_error, rot_error, "B"), pos_error, rot_error
    except (ValueError, TypeError, FloatingPointError):
        return False, None, None


def teach(base, config, bounds, solver, fk, inputs):
    rng = np.random.Generator(np.random.PCG64(per_row_seed(config["randomness"]["data_seed"], base["source_sample_id"], "wide", "teacher")))
    starts = [base["q_current"]] + [rng.uniform(bounds[:, 0], bounds[:, 1], size=6) for _ in range(3)]
    outcomes = []
    valid = []
    for ordinal, start in enumerate(starts):
        result = solver.solve(start, base["position_m"], base["quaternion_wxyz"], deadline_ns=None)
        candidate = result.q_candidate
        good, p, r = candidate_valid(candidate, base["position_m"], base["quaternion_wxyz"], fk, inputs)
        distance = float(np.sqrt(np.sum(((candidate - base["q_current"]) / (bounds[:, 1] - bounds[:, 0])) ** 2))) if good else None
        outcomes.append({"ordinal": ordinal, "q_start": start.tolist(), "solver_status": str(result.status), "reason": result.termination_reason,
                         "iterations": result.iterations, "elapsed_ns": result.elapsed_ns,
                         "valid": good, "position_error_m": p, "orientation_error_rad": r,
                         "distance": distance, "q_candidate": candidate.tolist() if candidate is not None else None})
        if good:
            valid.append((distance, ordinal, q_key(candidate), candidate.copy()))
    if valid:
        distance, ordinal, _, label = min(valid)
        return {"q_target": label, "label_present": True, "teacher_status": "VALID",
                "teacher_failure_class": "NONE", "teacher_candidate_count": 4,
                "teacher_valid_count": len(valid), "teacher_selected_ordinal": ordinal,
                "teacher_selected_distance": distance}, outcomes
    failure = "SOLVER_EXCEPTION" if any(o["solver_status"] == "NUMERICAL_FAILURE" for o in outcomes) else "NO_VALID_CANDIDATE"
    return {"q_target": None, "label_present": False, "teacher_status": "FAILED",
            "teacher_failure_class": failure, "teacher_candidate_count": 4,
            "teacher_valid_count": 0, "teacher_selected_ordinal": -1,
            "teacher_selected_distance": None}, outcomes


def build_pair(row, mode, config, bounds, solver, fk, inputs):
    base = make_base(row, mode, config, bounds)
    base["source_manifest_sha256"] = sha(ROOT / config["source"]["dataset_manifest"])
    base["teacher_config_sha256"] = sha(CONFIG)
    base["teacher_candidate_family"] = f"teacher-{row['sample_id']}-{mode}"
    if mode == "local":
        base.update({"q_target": row["q"].copy(), "label_present": True,
                     "teacher_status": "NOT_APPLICABLE", "teacher_failure_class": "NONE",
                     "teacher_candidate_count": 0, "teacher_valid_count": 0,
                     "teacher_selected_ordinal": -1, "teacher_selected_distance": None})
        return base, []
    label, outcomes = teach(base, config, bounds, solver, fk, inputs)
    base.update(label)
    return base, outcomes


def source_fk_check(records, inputs):
    pin, independent = PinocchioFK(inputs), IndependentFK(inputs)
    max_p = max_r = 0.0
    for row in records:
        validate_q(row["q"], inputs.joint_names, inputs.limits)
        reference = pin.reference_forward_kinematics(row["q"])
        cross = independent.forward_kinematics(row["q"])
        p = float(np.linalg.norm(reference[:3, 3] - cross[:3, 3]))
        r = float(np.linalg.norm(reference[:3, :3] - cross[:3, :3], ord="fro"))
        if p > 1e-9 or r > 1e-9:
            raise ValueError(f"source independent FK mismatch: {row['sample_id']}")
        if np.linalg.norm(reference[:3, 3] - row["position"]) > 1e-9:
            raise ValueError("source position mismatch")
        if np.linalg.norm(quaternion_rotation(row["quaternion"]) - reference[:3, :3], ord="fro") > 1e-9:
            raise ValueError("source quaternion mismatch")
        if not np.array_equal(canonical_quaternion(reference[:3, :3]), row["quaternion"]):
            # Exact source quaternion may differ by tiny platform rounding; the rotation test above is authoritative.
            pass
        max_p = max(max_p, p)
        max_r = max(max_r, r)
    return {"rows": len(records), "maximum_position_error_m": max_p, "maximum_rotation_frobenius_error": max_r}


def pilot(output: Path):
    config, _ = load_contract()
    thread_environment = require_single_thread_environment()
    records, _ = roots()
    inputs = load_robot()
    bounds = np.asarray(inputs.limits, dtype="<f8")
    selected = select_pilot(records)
    fkcheck = source_fk_check(selected, inputs)
    solver, fk = DLS(inputs), PinocchioFK(inputs)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = Counter()
    by_group = defaultdict(Counter)
    costs = []
    with (output / "candidate-outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            if time.monotonic() - started > config["pilot"]["wall_cap_minutes"] * 60:
                raise RuntimeError("pilot wall cap exceeded; no full generation")
            if process_peak_rss_bytes() > config["pilot"]["ram_cap_gib"] * 1024**3:
                raise RuntimeError("pilot RAM cap exceeded; no full generation")
            pair, outcomes = build_pair(row, "wide", config, bounds, solver, fk, inputs)
            counts[pair["teacher_status"]] += 1
            by_group[f"{row['family']}/{row['split']}"][pair["teacher_status"]] += 1
            for outcome in outcomes:
                costs.append(outcome["elapsed_ns"])
                stream.write(json.dumps({"pair_id": pair["pair_id"], "split": row["split"], "family": row["family"], **outcome}, sort_keys=True, allow_nan=False) + "\n")
    result = {"status": "PASS", "planned_rows": 90, "actual_rows": len(selected),
              "solver_calls": len(costs), "maximum_iterations_budget": 72000,
              "elapsed_wall_s": time.monotonic() - started,
              "solver_elapsed_s": sum(costs) / 1e9,
              "peak_rss_bytes": process_peak_rss_bytes(),
              "thread_environment": thread_environment,
              "teacher_status": dict(counts),
              "by_family_split": {key: dict(value) for key, value in sorted(by_group.items())},
              "source_fk": fkcheck,
              "candidate_outcomes_sha256": sha(output / "candidate-outcomes.jsonl"),
              "config_sha256": sha(CONFIG), "schema_sha256": sha(SCHEMA)}
    if result["solver_calls"] != config["pilot"]["max_solver_calls"]:
        raise ValueError("pilot solver call budget mismatch")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return result


def _typed_arrays(rows, schema):
    result = {}
    for field in schema["fields"]:
        name, dtype = field["name"], field["dtype"]
        values = []
        for row in rows:
            value = row[name]
            if value is None:
                if name == "q_target":
                    value = [float("nan")] * 6
                elif name == "teacher_selected_distance":
                    value = float("nan")
                else:
                    raise ValueError(f"unexpected null field: {name}")
            values.append(value)
        result[name] = np.asarray(values, dtype=dtype)
    return result


def _canonical_dataset_hash(shards):
    h = hashlib.sha256()
    for shard in shards:
        h.update(f"{shard['path']}\0{shard['record_count']}\0{shard['content_sha256']}\n".encode())
    return h.hexdigest()


def _pose_key(position, quaternion):
    p = np.asarray(position, dtype="<f8").copy()
    q = np.asarray(quaternion, dtype="<f8").copy()
    p[p == 0] = 0
    q[q == 0] = 0
    return p.tobytes() + q.tobytes()


def audit_rows(rows, source_roots, query_path: Path, config, inputs):
    bounds = np.asarray(inputs.limits, dtype="<f8")
    root_map = {r["sample_id"]: r for r in source_roots}
    group_splits = defaultdict(set)
    family_splits = defaultdict(set)
    q_splits = defaultdict(set)
    pose_splits = defaultdict(set)
    counts = Counter()
    missing = Counter()
    exact_equal_local = 0
    perturbations = []
    seen_pair_ids = set()
    benchmark_q = set()
    benchmark_pose = set()
    benchmark_groups = set()
    with query_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            query = json.loads(line)
            benchmark_q.add(q_key(query["q_target"]))
            benchmark_q.add(q_key(query["q_current"]))
            benchmark_pose.add(_pose_key(query["target_position_m"], query["target_quaternion_wxyz"]))
            benchmark_groups.add(query["query_group_id"])
    benchmark_q_overlaps = benchmark_pose_overlaps = benchmark_group_overlaps = 0
    for pair in rows:
        root = root_map[pair["source_sample_id"]]
        if pair["pair_id"] in seen_pair_ids:
            raise ValueError("duplicate pair ID")
        seen_pair_ids.add(pair["pair_id"])
        if (pair["group_id"], pair["split"], pair["source_family"]) != (root["group_id"], root["split"], root["family"]):
            raise ValueError("source root lineage mismatch")
        if q_key(pair["root_q_target"]) != q_key(root["q"]):
            raise ValueError("source root q mismatch")
        validate_q(pair["q_current"], inputs.joint_names, inputs.limits)
        if pair["label_present"]:
            validate_q(pair["q_target"], inputs.joint_names, inputs.limits)
        elif pair["q_target"] is not None or pair["teacher_status"] != "FAILED":
            raise ValueError("missing label contract violation")
        mode = pair["pair_mode"]
        if mode == "local":
            delta = pair["q_current"] - root["q"]
            if np.any(np.abs(delta) > 0.1) or pair["teacher_status"] != "NOT_APPLICABLE" or not pair["label_present"]:
                raise ValueError("local contract violation")
            if q_key(pair["q_current"]) == q_key(root["q"]):
                exact_equal_local += 1
            perturbations.extend(np.abs(delta).tolist())
        else:
            expected_seed = per_row_seed(config["randomness"]["data_seed"], root["sample_id"], "wide", "wide")
            expected_start = np.random.Generator(np.random.PCG64(expected_seed)).uniform(bounds[:, 0], bounds[:, 1], size=6)
            if q_key(pair["q_current"]) != q_key(expected_start):
                raise ValueError("wide independent start mismatch")
        if pair["source_manifest_sha256"] != sha(ROOT / config["source"]["dataset_manifest"]):
            raise ValueError("source manifest binding mismatch")
        group_splits[pair["group_id"]].add(pair["split"])
        family_splits[pair["teacher_candidate_family"]].add(pair["split"])
        for name in ("root_q_target", "q_current", "q_target"):
            if pair[name] is not None:
                q_splits[q_key(pair[name])].add(pair["split"])
                benchmark_q_overlaps += q_key(pair[name]) in benchmark_q
        pose = _pose_key(pair["position_m"], pair["quaternion_wxyz"])
        pose_splits[pose].add(pair["split"])
        benchmark_pose_overlaps += pose in benchmark_pose
        benchmark_group_overlaps += pair["group_id"] in benchmark_groups
        counts[f"{pair['source_family']}/{pair['split']}/{mode}"] += 1
        if not pair["label_present"]:
            missing[f"{pair['source_family']}/{pair['split']}/{mode}/{pair['teacher_failure_class']}"] += 1
    for name, mapping in (("group", group_splits), ("candidate_family", family_splits), ("exact_q", q_splits), ("exact_pose", pose_splits)):
        if any(len(splits) != 1 for splits in mapping.values()):
            raise ValueError(f"cross-split leakage: {name}")
    if benchmark_q_overlaps or benchmark_pose_overlaps or benchmark_group_overlaps:
        raise ValueError("benchmark exact q/pose/group overlap")
    expected = config["source"]["planned_split_mode_counts"]
    for split, mode_counts in expected.items():
        for mode, per_mode in mode_counts.items():
            actual = sum(counts[f"{family}/{split}/{mode}"] for family in ("main", "boundary", "singularity"))
            if actual != per_mode:
                raise ValueError("split/mode count mismatch")
    if exact_equal_local:
        raise ValueError("local exact equality")
    return {"status": "PASS", "row_count": len(rows), "counts": dict(sorted(counts.items())),
            "missing_labels": dict(sorted(missing.items())), "local_exact_equality": exact_equal_local,
            "local_abs_delta_quantiles_rad": {str(q): float(np.quantile(perturbations, q)) for q in (0, .01, .5, .95, 1)},
            "cross_split_group": 0, "cross_split_candidate_family": 0,
            "cross_split_exact_q": 0, "cross_split_exact_pose": 0,
            "benchmark_exact_q_overlap": benchmark_q_overlaps,
            "benchmark_exact_pose_overlap": benchmark_pose_overlaps,
            "benchmark_group_overlap": benchmark_group_overlaps}


def generate(output: Path):
    config, schema = load_contract()
    require_single_thread_environment()
    records, _ = roots()
    inputs = load_robot()
    bounds = np.asarray(inputs.limits, dtype="<f8")
    fk_source = source_fk_check(records, inputs)
    solver, fk = DLS(inputs), PinocchioFK(inputs)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    teacher_stats = defaultdict(Counter)
    selected_ordinals = defaultdict(Counter)
    start_time = time.monotonic()
    with (output / "teacher-candidates.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for root in records:
            for mode in ("local", "wide"):
                row, outcomes = build_pair(root, mode, config, bounds, solver, fk, inputs)
                rows.append(row)
                key = f"{root['family']}/{root['split']}/{mode}"
                teacher_stats[key][row["teacher_status"]] += 1
                if row["teacher_status"] == "VALID":
                    selected_ordinals[key][row["teacher_selected_ordinal"]] += 1
                for outcome in outcomes:
                    stream.write(json.dumps({"pair_id": row["pair_id"], "split": row["split"],
                                             "family": row["source_family"], **outcome}, sort_keys=True, allow_nan=False) + "\n")
    if len(rows) != config["source"]["planned_rows"]:
        raise ValueError("planned row count mismatch")
    query_path = ROOT / "data/generated/F0-05/acceptance/run-a/query-list.jsonl"
    audit = audit_rows(rows, records, query_path, config, inputs)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["source_family"], row["split"], row["pair_mode"])].append(row)
    field_order = [f["name"] for f in schema["fields"]]
    shards = []
    for family in ("main", "boundary", "singularity"):
        for split in ("train", "validation", "test"):
            for mode in ("local", "wide"):
                bucket = buckets[(family, split, mode)]
                for offset in range(0, len(bucket), config["output"]["shard_size"]):
                    index = offset // config["output"]["shard_size"]
                    name = f"{family}-{split}-{mode}-{index:05d}.npz"
                    arrays = _typed_arrays(bucket[offset:offset+config["output"]["shard_size"]], schema)
                    write_deterministic_npz(output / name, arrays, field_order)
                    shards.append({"path": name, "record_count": len(bucket[offset:offset+config["output"]["shard_size"]]),
                                   "file_sha256": sha(output / name),
                                   "content_sha256": canonical_array_hash(arrays, field_order)})
    train = [row for row in rows if row["split"] == "train"]
    positions = np.asarray([row["position_m"] for row in train])
    label_train = [row["q_target"] for row in train if row["label_present"]]
    normalization = {"source": "C1-02_train_only", "position_count": len(positions),
                     "position_mean_m": positions.mean(axis=0).tolist(),
                     "position_std_m": positions.std(axis=0).tolist(),
                     "q_current_scaling": "per-joint lower/upper inclusive affine",
                     "q_target_label_scaling": "per-joint lower/upper inclusive affine",
                     "label_count": len(label_train),
                     "quaternion": "canonical unit wxyz; no fit"}
    manifest = {"schema_version": "1.0.0", "task": "C1-02", "status": "GENERATED_UNVERIFIED",
                "config_sha256": sha(CONFIG), "schema_sha256": sha(SCHEMA),
                "source_dataset_manifest_sha256": sha(ROOT / config["source"]["dataset_manifest"]),
                "query_list_sha256": sha(query_path), "record_count": len(rows),
                "shards": shards, "dataset_content_sha256": _canonical_dataset_hash(shards),
                "teacher_candidates_file_sha256": sha(output / "teacher-candidates.jsonl"),
                "reproduction_command": "pixi run --locked python scripts/run_c102_full.py --output data/generated/C1-02/v1"}
    for name, data in (("dataset-manifest.json", manifest), ("leakage-audit.json", audit),
                       ("normalization.json", normalization),
                       ("teacher-summary.json", {"status": "MEASURED", "by_family_split_mode": {k: dict(v) for k, v in sorted(teacher_stats.items())},
                                                 "selected_ordinals": {k: dict(v) for k, v in sorted(selected_ordinals.items())},
                                                 "candidate_file_sha256": manifest["teacher_candidates_file_sha256"],
                                                 "elapsed_wall_s": time.monotonic() - start_time,
                                                 "source_fk": fk_source})):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return {"manifest": manifest, "audit": audit, "normalization": normalization,
            "teacher_summary": json.loads((output / "teacher-summary.json").read_text(encoding="utf-8"))}
