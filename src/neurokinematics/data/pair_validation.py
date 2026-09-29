"""Independent C1-02 shard, label, lineage and benchmark acceptance audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from neurokinematics.benchmark.queries import q_key
from neurokinematics.data.factory import canonical_array_hash, read_shard
from neurokinematics.data.pairs import (CONFIG, SCHEMA, ROOT, _canonical_dataset_hash,
    _pose_key, audit_rows, candidate_valid, load_contract, per_row_seed, roots, sha)
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import load_robot, validate_q
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


def checked_input_projection(row):
    config, _ = load_contract()
    fields = config["normalization"]["input_fields"]
    forbidden = set(config["normalization"]["forbidden_input_fields"])
    if set(fields) & forbidden or fields != ["position_m", "quaternion_wxyz", "q_current"]:
        raise ValueError("q_target or provenance input leakage")
    return {name: row[name] for name in fields}


def validate_one(row, root, inputs, fk):
    if row["source_sample_id"] != root["sample_id"] or row["group_id"] != root["group_id"] or row["split"] != root["split"]:
        raise ValueError("root split or lineage mismatch")
    if not np.isfinite(row["q_current"]).all() or not np.isfinite(row["position_m"]).all() or not np.isfinite(row["quaternion_wxyz"]).all():
        raise ValueError("nonfinite input")
    validate_q(row["q_current"], inputs.joint_names, inputs.limits)
    quat = row["quaternion_wxyz"]
    if abs(float(np.linalg.norm(quat)) - 1) > 1e-12 or quat[0] < 0 or (quat[0] == 0 and next((x for x in quat[1:] if x != 0), 1) < 0):
        raise ValueError("noncanonical quaternion")
    if np.linalg.norm(row["position_m"] - root["position"]) > 1e-9 or np.linalg.norm(quaternion_rotation(quat) - quaternion_rotation(root["quaternion"]), ord="fro") > 1e-9:
        raise ValueError("target pose differs from source")
    if row["pair_mode"] == "local":
        if row["teacher_status"] != "NOT_APPLICABLE" or not row["label_present"] or q_key(row["q_target"]) != q_key(root["q"]):
            raise ValueError("local label contract violation")
        if np.any(np.abs(row["q_current"] - root["q"]) > 0.1) or q_key(row["q_current"]) == q_key(root["q"]):
            raise ValueError("local perturbation violation")
    elif row["pair_mode"] == "wide":
        if row["label_present"] != (row["teacher_status"] == "VALID"):
            raise ValueError("wide label status mismatch")
    else:
        raise ValueError("pair_mode violation")
    if row["label_present"]:
        good, _, _ = candidate_valid(row["q_target"], row["position_m"], quat, fk, inputs)
        if not good:
            raise ValueError("independent label FK failed")
    elif row["q_target"] is not None or row["teacher_failure_class"] == "NONE":
        raise ValueError("missing label failure class violation")
    checked_input_projection(row)


def load_rows(output: Path, manifest: dict, schema: dict):
    fields = schema["fields"]
    order = [field["name"] for field in fields]
    rows = []
    for shard in manifest["shards"]:
        path = output / shard["path"]
        if not path.is_file() or sha(path) != shard["file_sha256"]:
            raise ValueError(f"shard SHA mismatch: {path}")
        arrays = read_shard(path, order)
        for field in fields:
            name = field["name"]
            arr = arrays[name]
            if arr.dtype.str != np.dtype(field["dtype"]).str or arr.shape != (shard["record_count"], *field["shape_per_record"]):
                raise ValueError(f"schema dtype/shape mismatch: {name}")
            if arr.dtype.kind == "f" and name not in ("q_target", "teacher_selected_distance") and not np.isfinite(arr).all():
                raise ValueError(f"nonfinite field: {name}")
        if canonical_array_hash(arrays, order) != shard["content_sha256"]:
            raise ValueError("shard content hash mismatch")
        for index in range(shard["record_count"]):
            row = {}
            for field in fields:
                value = arrays[field["name"]][index]
                if field["dtype"].startswith("|S"):
                    value = value.decode()
                elif field["shape_per_record"]:
                    value = value.astype("<f8")
                elif field["dtype"] == "|b1":
                    value = bool(value)
                elif field["dtype"].startswith("<f"):
                    value = float(value)
                else:
                    value = int(value)
                row[field["name"]] = value
            if row["label_present"]:
                if not np.isfinite(row["q_target"]).all():
                    raise ValueError("present label NaN")
            elif not np.isnan(row["q_target"]).all():
                raise ValueError("missing label sentinel mismatch")
            if not row["label_present"]:
                row["q_target"] = None
            if np.isnan(row["teacher_selected_distance"]):
                if row["pair_mode"] != "local" and row["label_present"]:
                    raise ValueError("selected distance missing")
                row["teacher_selected_distance"] = None
            elif row["pair_mode"] == "local" or not row["label_present"]:
                raise ValueError("unexpected selected distance")
            rows.append(row)
    if len(rows) != manifest["record_count"]:
        raise ValueError("manifest row count mismatch")
    return rows


def _near_pose_audit(roots_data):
    coordinates = np.asarray([row["position"] for row in roots_data])
    tree = cKDTree(coordinates)
    suspects = tree.query_pairs(r=1e-9, output_type="ndarray")
    collisions = 0
    for a, b in suspects:
        if roots_data[a]["split"] == roots_data[b]["split"]:
            continue
        rotation_a = quaternion_rotation(roots_data[a]["quaternion"])
        rotation_b = quaternion_rotation(roots_data[b]["quaternion"])
        if rotation_error(rotation_a, rotation_b) <= 1e-9:
            collisions += 1
    if collisions:
        raise ValueError("cross-split semantically same target pose")
    return {"position_near_pairs_checked": len(suspects), "cross_split_near_pose": collisions}


def _benchmark_near_pose_audit(roots_data, query_path):
    queries = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
    coordinates = np.asarray([row["target_position_m"] for row in queries])
    tree = cKDTree(coordinates)
    suspects = 0
    for root in roots_data:
        for index in tree.query_ball_point(root["position"], r=1e-9):
            suspects += 1
            if rotation_error(quaternion_rotation(root["quaternion"]),
                              quaternion_rotation(queries[index]["target_quaternion_wxyz"])) <= 1e-9:
                raise ValueError("benchmark semantically same target pose")
    return {"position_near_pairs_checked": suspects, "same_target_pose": 0}


def _candidate_audit(output, rows, inputs, fk, expected_sha):
    path = output / "teacher-candidates.jsonl"
    if sha(path) != expected_sha:
        raise ValueError("candidate log SHA mismatch")
    by_id = {row["pair_id"]: row for row in rows if row["pair_mode"] == "wide"}
    seen = defaultdict(list)
    split_by_q = defaultdict(set)
    discarded_candidate_splits = defaultdict(set)
    status = Counter()
    valid_by_ordinal = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            item = json.loads(raw)
            pair = by_id.get(item["pair_id"])
            if pair is None or item["split"] != pair["split"] or item["family"] != pair["source_family"]:
                raise ValueError("candidate lineage mismatch")
            start = validate_q(item["q_start"], inputs.joint_names, inputs.limits)
            split_by_q[q_key(start)].add(pair["split"])
            candidate = item["q_candidate"]
            if candidate is not None:
                candidate = np.asarray(candidate, dtype="<f8")
            valid, _, _ = candidate_valid(candidate, pair["position_m"], pair["quaternion_wxyz"], fk, inputs)
            if bool(item["valid"]) != valid:
                raise ValueError("candidate independent FK verdict mismatch")
            if valid:
                split_by_q[q_key(candidate)].add(pair["split"])
                bounds = np.asarray(inputs.limits)
                distance = float(np.sqrt(np.sum(((candidate - pair["q_current"]) / (bounds[:, 1] - bounds[:, 0])) ** 2)))
                if distance != item["distance"]:
                    raise ValueError("candidate distance mismatch")
                valid_by_ordinal[item["ordinal"]] += 1
            elif candidate is not None and candidate.shape == (6,) and np.isfinite(candidate).all():
                # Failed solver iterates are neither model input nor label. Retain overlap as bias diagnostic.
                discarded_candidate_splits[q_key(candidate)].add(pair["split"])
            status[item["solver_status"]] += 1
            seen[item["pair_id"]].append((item["ordinal"], valid, item["distance"], candidate))
    if len(seen) != len(by_id):
        raise ValueError("candidate inventory incomplete")
    for pair_id, outcomes in seen.items():
        pair = by_id[pair_id]
        if len(outcomes) != 4 or sorted(x[0] for x in outcomes) != [0, 1, 2, 3]:
            raise ValueError("candidate budget/ordinal mismatch")
        valid = [(distance, ordinal, q_key(candidate), candidate) for ordinal, good, distance, candidate in outcomes if good]
        if len(valid) != pair["teacher_valid_count"] or pair["teacher_candidate_count"] != 4:
            raise ValueError("candidate count mismatch")
        if valid:
            distance, ordinal, _, chosen = min(valid)
            if (pair["teacher_status"] != "VALID" or pair["teacher_selected_ordinal"] != ordinal or
                    pair["teacher_selected_distance"] != distance or q_key(pair["q_target"]) != q_key(chosen)):
                raise ValueError("teacher selected wrong valid branch")
        elif pair["teacher_status"] != "FAILED" or pair["label_present"]:
            raise ValueError("failed teacher row was filtered or labelled")
    if any(len(s) > 1 for s in split_by_q.values()):
        raise ValueError("cross-split teacher restart/candidate exact q")
    return {"status": "PASS", "wide_rows": len(by_id), "candidate_rows": sum(status.values()),
            "solver_status": dict(status), "valid_candidate_by_ordinal": dict(valid_by_ordinal),
            "cross_split_restart_or_valid_candidate_q": 0,
            "discarded_invalid_candidate_q_cross_split_keys": sum(len(splits) > 1 for splits in discarded_candidate_splits.values())}


def validate_normalization(normalization, rows):
    family_order = {"main": 0, "boundary": 1, "singularity": 2}
    mode_order = {"local": 0, "wide": 1}
    train_rows = sorted((row for row in rows if row["split"] == "train"),
                        key=lambda row: (family_order[row["source_family"]], row["source_sample_id"], mode_order[row["pair_mode"]]))
    train_positions = np.asarray([row["position_m"] for row in train_rows])
    if (normalization["source"] != "C1-02_train_only" or normalization["position_count"] != len(train_positions) or
            normalization["position_mean_m"] != train_positions.mean(axis=0).tolist() or
            normalization["position_std_m"] != train_positions.std(axis=0).tolist() or
            normalization["label_count"] != sum(row["split"] == "train" and row["label_present"] for row in rows)):
        raise ValueError("normalization train-only violation")


def _ancestry_audit(rows):
    source_splits = defaultdict(set)
    seed_splits = defaultdict(set)
    for row in rows:
        source_splits[row["source_sample_id"]].add(row["split"])
        seed_splits[row["derivation_seed"]].add(row["split"])
    if any(len(splits) > 1 for splits in source_splits.values()):
        raise ValueError("source root lineage crossed splits")
    if any(len(splits) > 1 for splits in seed_splits.values()):
        raise ValueError("resampling seed crossed splits")
    return {"source_roots": len(source_splits), "derivation_seeds": len(seed_splits),
            "cross_split_source_root": 0, "cross_split_resampling_seed": 0}


def verify(output: Path):
    output = Path(output)
    config, schema = load_contract()
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha(CONFIG) or manifest["schema_sha256"] != sha(SCHEMA):
        raise ValueError("config/schema SHA mismatch")
    if manifest["source_dataset_manifest_sha256"] != sha(ROOT / config["source"]["dataset_manifest"]):
        raise ValueError("source manifest SHA mismatch")
    if manifest["dataset_content_sha256"] != _canonical_dataset_hash(manifest["shards"]):
        raise ValueError("dataset content SHA mismatch")
    roots_data, _ = roots()
    inputs = load_robot()
    fk = PinocchioFK(inputs)
    rows = load_rows(output, manifest, schema)
    root_map = {root["sample_id"]: root for root in roots_data}
    for row in rows:
        validate_one(row, root_map[row["source_sample_id"]], inputs, fk)
    query_path = ROOT / "data/generated/F0-05/acceptance/run-a/query-list.jsonl"
    if manifest["query_list_sha256"] != sha(query_path):
        raise ValueError("benchmark query SHA mismatch")
    leakage = audit_rows(rows, roots_data, query_path, config, inputs)
    ancestry = _ancestry_audit(rows)
    near_pose = _near_pose_audit(roots_data)
    benchmark_near_pose = _benchmark_near_pose_audit(roots_data, query_path)
    candidate = _candidate_audit(output, rows, inputs, fk, manifest["teacher_candidates_file_sha256"])
    normalization = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    validate_normalization(normalization, rows)
    result = {"status": "PASS", "records": len(rows), "shards": len(manifest["shards"]),
              "dataset_content_sha256": manifest["dataset_content_sha256"],
              "leakage": leakage, "ancestry": ancestry, "near_pose": near_pose,
              "benchmark_near_pose": benchmark_near_pose,
              "candidate": candidate,
              "normalization_train_only": "PASS"}
    return result
