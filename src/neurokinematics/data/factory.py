"""F0-04 deterministic data factory.

The frozen JSON config is the only source of sampling and coverage decisions.
No module-global RNG is used.  Shards are deterministic ZIP containers whose
members are uncompressed NPY arrays with fixed timestamps and member order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.metrics import quaternion_rotation, singularity_metrics
from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK

CONFIG_PATH = ROOT / "experiments/F0-04/config.json"
SCHEMA_PATH = ROOT / "experiments/F0-04/schema.json"
FIELD_DTYPES = {
    "robot_id": "|S32", "model_hash": "|S64", "tcp_hash": "|S64",
    "sample_id": "|S40", "group_id": "|S40", "split": "|S10",
    "q": "<f8", "position_m": "<f8", "quaternion_wxyz": "<f8",
    "sampling_class": "|S16", "sigma_min": "<f8", "sigma_max": "<f8",
    "condition": "<f8", "manipulability": "<f8", "numerical_rank": "|i1",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_hash(path: Path) -> str:
    # Frozen artifacts are identified by their exact bytes, not a reserialization.
    return sha256_file(path)


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def latin_hypercube(count: int, limits, seed: int) -> np.ndarray:
    """One jittered point per stratum per dimension; independent PCG64 permutations."""
    bounds = np.asarray(limits, dtype=np.float64)
    generator = _rng(seed)
    unit = np.empty((count, bounds.shape[0]), dtype=np.float64)
    for dimension in range(bounds.shape[0]):
        permutation = generator.permutation(count)
        unit[:, dimension] = (permutation + generator.random(count)) / count
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def uniform_samples(count: int, limits, seed: int) -> np.ndarray:
    bounds = np.asarray(limits, dtype=np.float64)
    return _rng(seed).uniform(bounds[:, 0], bounds[:, 1], size=(count, len(bounds)))


def boundary_samples(count: int, limits, seed: int, threshold: float) -> tuple[np.ndarray, list[dict]]:
    bounds = np.asarray(limits, dtype=np.float64)
    generator = _rng(seed)
    q = generator.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(bounds)))
    labels = []
    for i in range(count):
        joint = (i // 2) % len(bounds)
        side = i % 2
        distance = generator.random() * threshold
        q[i, joint] = bounds[joint, side] + (distance if side == 0 else -distance) * (bounds[joint, 1] - bounds[joint, 0])
        labels.append({"joint_index": joint, "side": "lower" if side == 0 else "upper"})
    return q, labels


def canonical_quaternion(rotation) -> np.ndarray:
    """Matrix to normalized wxyz with a deterministic sign, including exact pi."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        result = np.array([0.25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(0.0, 1.0 + r[i,i] - r[j,j] - r[k,k]))
        xyz = np.empty(3, dtype=np.float64)
        xyz[i] = 0.25*s
        xyz[j] = (r[j,i] + r[i,j]) / s
        xyz[k] = (r[k,i] + r[i,k]) / s
        result = np.r_[(r[k,j] - r[j,k]) / s, xyz]
    result /= np.linalg.norm(result)
    # Treat only an exact zero scalar as the pi tie.  This avoids a tunable epsilon.
    if result[0] < 0 or (result[0] == 0 and next((v for v in result[1:] if v != 0), 1.0) < 0):
        result = -result
    return result.astype("<f8")


def assign_splits(group_ids: list[str], seed: int, ratios: dict[str, float]) -> np.ndarray:
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("duplicate group_id in independent-root dataset")
    count = len(group_ids)
    order = _rng(seed).permutation(count)
    cut_train = int(count * ratios["train"])
    cut_validation = cut_train + int(count * ratios["validation"])
    labels = np.empty(count, dtype="|S10")
    labels[order[:cut_train]] = b"train"
    labels[order[cut_train:cut_validation]] = b"validation"
    labels[order[cut_validation:]] = b"test"
    return labels


def inherited_split(root_group_ids: list[str], variant_group_ids: list[str], root_splits) -> np.ndarray:
    mapping = dict(zip(root_group_ids, np.asarray(root_splits, dtype="|S10"), strict=True))
    try:
        return np.asarray([mapping[group] for group in variant_group_ids], dtype="|S10")
    except KeyError as error:
        raise ValueError("variant has no declared root group") from error


def canonical_array_hash(arrays: dict[str, np.ndarray], field_order: list[str]) -> str:
    digest = hashlib.sha256()
    for name in field_order:
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        dtype = array.dtype.str.replace(">", "<")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(dtype.encode("ascii") + b"\0")
        digest.update(",".join(map(str, array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray], field_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for name in field_order:
            payload = io.BytesIO()
            np.save(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())


def read_shard(path: Path, field_order: list[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if sorted(loaded.files) != sorted(field_order):
                raise ValueError("shard field set mismatch")
            return {name: loaded[name] for name in field_order}
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid or truncated shard") from error


def validate_arrays(arrays: dict[str, np.ndarray], schema: dict, limits, expected_metadata=None) -> None:
    fields = schema["fields"]
    count = len(arrays[fields[0]["name"]])
    for field in fields:
        name = field["name"]
        if name not in arrays or arrays[name].dtype.str != field["dtype"]:
            raise ValueError(f"schema dtype mismatch: {name}")
        expected = (count, *field["shape_per_record"])
        if arrays[name].shape != expected:
            raise ValueError(f"schema shape mismatch: {name}")
        if arrays[name].dtype.kind == "f" and np.isnan(arrays[name]).any():
            raise ValueError(f"NaN forbidden: {name}")
        if arrays[name].dtype.kind == "f" and name != "condition" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"Inf forbidden: {name}")
    condition, sigma_min = arrays["condition"], arrays["sigma_min"]
    if np.any(np.isinf(condition) != (sigma_min == 0)):
        raise ValueError("condition Inf policy violation")
    if len(set(arrays["sample_id"].tolist())) != count:
        raise ValueError("duplicate sample_id")
    if expected_metadata:
        for name, expected in expected_metadata.items():
            if not np.all(arrays[name] == expected.encode()):
                raise ValueError(f"metadata mismatch: {name}")
    bounds = np.asarray(limits)
    if np.any(arrays["q"] < bounds[:, 0]) or np.any(arrays["q"] > bounds[:, 1]):
        raise ValueError("joint limit violation")
    if not set(arrays["split"].tolist()) <= {b"train", b"validation", b"test"}:
        raise ValueError("invalid split")
    norms = np.linalg.norm(arrays["quaternion_wxyz"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-12, rtol=0):
        raise ValueError("invalid quaternion norm")
    for quaternion in arrays["quaternion_wxyz"]:
        if quaternion[0] < 0 or (quaternion[0] == 0 and next((v for v in quaternion[1:] if v != 0), 1.0) < 0):
            raise ValueError("noncanonical quaternion")


def audit_groups_and_duplicates(arrays: dict[str, np.ndarray], independent_roots: bool = True) -> dict:
    groups = arrays["group_id"].tolist()
    if independent_roots and len(set(groups)) != len(groups):
        raise ValueError("duplicate group_id in independent-root dataset")
    group_splits = {}
    for group, split in zip(groups, arrays["split"].tolist(), strict=True):
        group_splits.setdefault(group, set()).add(split)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("same group occurs in multiple splits")
    seen = {}
    for row, split in zip(arrays["q"], arrays["split"].tolist(), strict=True):
        key = hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
        if key in seen and seen[key] != split:
            raise ValueError("same q occurs in multiple splits")
        seen[key] = split
    return {"group_count": len(group_splits), "q_count": len(seen)}


def validate_boundary(q, limits, threshold: float) -> None:
    bounds = np.asarray(limits, dtype=np.float64)
    normalized = np.minimum((q-bounds[:,0])/(bounds[:,1]-bounds[:,0]),
                            (bounds[:,1]-q)/(bounds[:,1]-bounds[:,0]))
    if not np.all(np.min(normalized, axis=1) < threshold):
        raise ValueError("boundary threshold violation")


def train_singularity_threshold(arrays, quantile: float) -> float:
    train = arrays["split"] == b"train"
    if not np.any(train): raise ValueError("empty train split")
    return float(np.quantile(arrays["sigma_min"][train], quantile, method="linear"))


def _identifiers(prefix: str, count: int) -> tuple[list[str], list[str]]:
    return ([f"{prefix}-{i:08d}" for i in range(count)],
            [f"root-{prefix}-{i:08d}" for i in range(count)])


@dataclass
class BuildResult:
    arrays: dict[str, np.ndarray]
    max_position_error: float
    max_rotation_error: float


def _records(q: np.ndarray, prefix: str, sampling_class: str, splits, inputs) -> BuildResult:
    reference, independent = PinocchioFK(inputs), IndependentFK(inputs)
    jacobian = IndependentJacobian(inputs)
    count = len(q)
    sample_ids, group_ids = _identifiers(prefix, count)
    positions = np.empty((count, 3), dtype="<f8")
    quaternions = np.empty((count, 4), dtype="<f8")
    metrics = {key: np.empty(count, dtype="<f8") for key in ("sigma_min", "sigma_max", "condition", "manipulability")}
    ranks = np.empty(count, dtype="|i1")
    max_position, max_rotation = 0.0, 0.0
    for i, configuration in enumerate(q):
        expected = reference.reference_forward_kinematics(configuration)
        actual = independent.forward_kinematics(configuration)
        position_error = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
        rotation_error = float(np.linalg.norm(expected[:3, :3] - actual[:3, :3], ord="fro"))
        max_position, max_rotation = max(max_position, position_error), max(max_rotation, rotation_error)
        positions[i], quaternions[i] = expected[:3, 3], canonical_quaternion(expected[:3, :3])
        if np.linalg.norm(quaternion_rotation(quaternions[i]) - expected[:3, :3], ord="fro") > 1e-12:
            raise ValueError("quaternion reconstruction mismatch")
        values = singularity_metrics(jacobian.jacobian(configuration), 0.9015)
        for key in metrics:
            metrics[key][i] = values[key]
        ranks[i] = values["numerical_rank"]
    arrays = {
        "robot_id": np.full(count, inputs.robot_id.encode(), dtype="|S32"),
        "model_hash": np.full(count, inputs.hashes["assets/robots/robot_a/robot.urdf"].encode(), dtype="|S64"),
        "tcp_hash": np.full(count, inputs.hashes["config/robots/tcp_tool0.json"].encode(), dtype="|S64"),
        "sample_id": np.asarray(sample_ids, dtype="|S40"), "group_id": np.asarray(group_ids, dtype="|S40"),
        "split": np.asarray(splits, dtype="|S10"), "q": np.asarray(q, dtype="<f8"),
        "position_m": positions, "quaternion_wxyz": quaternions,
        "sampling_class": np.full(count, sampling_class.encode(), dtype="|S16"),
        **metrics, "numerical_rank": ranks,
    }
    return BuildResult(arrays, max_position, max_rotation)


def _q_hashes(q: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest() for row in q]


def normalization(arrays: dict[str, np.ndarray]) -> dict:
    train = arrays["split"] == b"train"
    return {"source": "main_train_only", "count": int(train.sum()),
            "q_mean": arrays["q"][train].mean(axis=0).tolist(),
            "q_std": arrays["q"][train].std(axis=0).tolist(),
            "position_mean_m": arrays["position_m"][train].mean(axis=0).tolist(),
            "position_std_m": arrays["position_m"][train].std(axis=0).tolist()}


def _rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray([quaternion_rotation(q) for q in quaternions])


def coverage(q, positions, quaternions, limits, profile, prefixes) -> dict:
    bounds = np.asarray(limits)
    normalized = (q - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    rotations = _rotation_matrices(quaternions)
    joint_keys = np.floor(normalized / profile["joint_space_bin_width_normalized"]).astype(np.int64)
    position_keys = np.floor(positions / profile["position_voxel_size_m"]).astype(np.int64)
    step = profile.get("orientation_matrix_step", profile.get("orientation_bin", {}).get("step"))
    orientation_keys = np.floor((rotations.reshape(len(q), 9) + 1.0) / step).astype(np.int64)
    max_bin = int(np.floor(2.0 / step))
    orientation_keys = np.clip(orientation_keys, 0, max_bin)
    def unique_count(values): return len({tuple(row) for row in values.tolist()})
    curve = []
    for prefix in prefixes:
        combined = np.concatenate((position_keys[:prefix], orientation_keys[:prefix]), axis=1)
        curve.append({"count": prefix, "joint": unique_count(joint_keys[:prefix]),
                      "position": unique_count(position_keys[:prefix]), "orientation": unique_count(orientation_keys[:prefix]),
                      "pose": unique_count(combined)})
    position_counts = Counter(map(tuple, position_keys.tolist()))
    return {"joint_space_bin_occupancy": unique_count(joint_keys),
            "position_voxel_occupancy": unique_count(position_keys),
            "orientation_bin_occupancy": unique_count(orientation_keys),
            "combined_pose_bin_occupancy": unique_count(np.concatenate((position_keys, orientation_keys), axis=1)),
            "prefix_saturation": curve,
            "densest_position_voxels": [{"key": list(key), "count": count} for key, count in position_counts.most_common(5)],
            "sparsest_position_voxel_count": min(position_counts.values()),
            "definitions": profile}


def _pose_only(q, inputs):
    reference = PinocchioFK(inputs)
    positions, quaternions = np.empty((len(q), 3)), np.empty((len(q), 4))
    for i, row in enumerate(q):
        pose = reference.reference_forward_kinematics(row)
        positions[i], quaternions[i] = pose[:3, 3], canonical_quaternion(pose[:3, :3])
    return positions, quaternions


def _write_shards(output: Path, name: str, arrays, shard_size, field_order):
    entries = []
    for index, start in enumerate(range(0, len(arrays["q"]), shard_size)):
        shard = {key: value[start:start+shard_size] for key, value in arrays.items()}
        path = output / f"{name}-{index:05d}.npz"
        write_deterministic_npz(path, shard, field_order)
        entries.append({"index": index, "path": path.name, "record_count": len(shard["q"]),
                        "file_sha256": sha256_file(path),
                        "content_sha256": canonical_array_hash(shard, field_order),
                        "fields": {key: {"dtype": value.dtype.str, "shape": list(value.shape)} for key, value in shard.items()}})
    return entries


def _dataset_hash(shards):
    digest = hashlib.sha256()
    for subset in ("main", "boundary", "singularity"):
        for item in shards[subset]:
            digest.update(f"{subset}\0{item['index']}\0{item['record_count']}\0{item['content_sha256']}\n".encode())
    return digest.hexdigest()


def generate_dataset(output: Path, evidence: Path | None = None, config_path: Path = CONFIG_PATH,
                     schema_path: Path = SCHEMA_PATH) -> dict:
    output, config_path, schema_path = Path(output), Path(config_path), Path(schema_path)
    output.mkdir(parents=True, exist_ok=True)
    config, schema, inputs = _json(config_path), _json(schema_path), load_robot()
    field_order = [field["name"] for field in schema["fields"]]
    if config["bit_generator"] != "PCG64" or config["dtype"] != "float64":
        raise ValueError("unsupported frozen RNG or dtype")
    limits = inputs.limits
    main_q = latin_hypercube(config["main"]["count"], limits, config["main_seed"])
    main_ids, main_groups = _identifiers("main", len(main_q))
    main_splits = assign_splits(main_groups, config["derived_seeds"]["split"], config["split"]["ratios"])
    main = _records(main_q, "main", "main_lhs", main_splits, inputs)
    threshold = train_singularity_threshold(main.arrays, config["hard_subsets"]["singularity"]["threshold_quantile"])
    boundary_q, boundary_labels = boundary_samples(config["hard_subsets"]["boundary"]["count"], limits,
        config["derived_seeds"]["boundary"], config["hard_subsets"]["boundary"]["normalized_joint_limit_distance_lt"])
    _, boundary_groups = _identifiers("boundary", len(boundary_q))
    boundary_splits = assign_splits(boundary_groups, config["derived_seeds"]["split"] + 1, config["split"]["ratios"])
    boundary = _records(boundary_q, "boundary", "boundary", boundary_splits, inputs)
    candidate_limit = config["hard_subsets"]["singularity"]["candidate_limit"]
    generator = _rng(config["derived_seeds"]["singularity_candidates"])
    bounds = np.asarray(limits)
    jacobian = IndependentJacobian(inputs)
    accepted, scanned = [], 0
    while len(accepted) < config["hard_subsets"]["singularity"]["count"] and scanned < candidate_limit:
        candidate = generator.uniform(bounds[:, 0], bounds[:, 1])
        scanned += 1
        if singularity_metrics(jacobian.jacobian(candidate), config["kinematics"]["characteristic_length_m"])["sigma_min"] <= threshold:
            accepted.append(candidate)
    if len(accepted) < config["hard_subsets"]["singularity"]["count"]:
        raise RuntimeError("candidate cap reached before singularity subset was complete")
    singular_q = np.asarray(accepted, dtype="<f8")
    _, singular_groups = _identifiers("singularity", len(singular_q))
    singular_splits = assign_splits(singular_groups, config["derived_seeds"]["split"] + 2, config["split"]["ratios"])
    singular = _records(singular_q, "singularity", "singularity", singular_splits, inputs)
    expected_metadata = {"robot_id": inputs.robot_id,
                         "model_hash": inputs.hashes["assets/robots/robot_a/robot.urdf"],
                         "tcp_hash": inputs.hashes["config/robots/tcp_tool0.json"]}
    for result in (main, boundary, singular):
        validate_arrays(result.arrays, schema, limits, expected_metadata)
        audit_groups_and_duplicates(result.arrays)
    validate_boundary(boundary_q, limits, config["hard_subsets"]["boundary"]["normalized_joint_limit_distance_lt"])
    position_limit, rotation_limit = config["kinematics"]["position_validation_atol_m"], config["kinematics"]["rotation_frobenius_validation_atol"]
    if max(x.max_position_error for x in (main, boundary, singular)) > position_limit or max(x.max_rotation_error for x in (main, boundary, singular)) > rotation_limit:
        raise ValueError("FK validation threshold exceeded")
    all_q_hashes = {name: set(_q_hashes(result.arrays["q"])) for name, result in (("main", main), ("boundary", boundary), ("singularity", singular))}
    overlap = {"main_boundary": len(all_q_hashes["main"] & all_q_hashes["boundary"]),
               "main_singularity": len(all_q_hashes["main"] & all_q_hashes["singularity"]),
               "boundary_singularity": len(all_q_hashes["boundary"] & all_q_hashes["singularity"])}
    if any(overlap.values()): raise ValueError("hard subset exact overlap")
    shards = {"main": _write_shards(output, "main", main.arrays, config["main"]["shard_size"], field_order),
              "boundary": _write_shards(output, "boundary", boundary.arrays, config["main"]["shard_size"], field_order),
              "singularity": _write_shards(output, "singularity", singular.arrays, config["main"]["shard_size"], field_order)}
    uniform_q = uniform_samples(config["uniform_comparison"]["count"], limits, config["derived_seeds"]["uniform_comparison"])
    uniform_position, uniform_quaternion = _pose_only(uniform_q, inputs)
    primary_profile = {"joint_space_bin_width_normalized": config["coverage"]["joint_space_bin_width_normalized"],
                       "position_voxel_size_m": config["coverage"]["position_voxel_size_m"],
                       "orientation_matrix_step": config["coverage"]["orientation_bin"]["step"]}
    coverage_summary = {"claim_scope": "empirical sampled-pool occupancy only; not a percentage of universally reachable workspace",
                        "main_lhs": coverage(main_q, main.arrays["position_m"], main.arrays["quaternion_wxyz"], limits, primary_profile, config["coverage"]["prefix_counts"]),
                        "uniform_comparison": coverage(uniform_q, uniform_position, uniform_quaternion, limits, primary_profile, config["coverage"]["prefix_counts"])}
    sensitivity = {profile["name"]: {"main_lhs": coverage(main_q, main.arrays["position_m"], main.arrays["quaternion_wxyz"], limits, profile, config["coverage"]["prefix_counts"]),
                                             "uniform_comparison": coverage(uniform_q, uniform_position, uniform_quaternion, limits, profile, config["coverage"]["prefix_counts"])}
                   for profile in config["coverage"]["sensitivity_profiles"]}
    split_counts = {name: {split: int(np.count_nonzero(result.arrays["split"] == split.encode())) for split in ("train", "validation", "test")}
                    for name, result in (("main", main), ("boundary", boundary), ("singularity", singular))}
    main_groups_by_split = {split: set(main.arrays["group_id"][main.arrays["split"] == split.encode()].tolist()) for split in ("train", "validation", "test")}
    split_audit = {"status": "PASS", "assignment_unit": "group_id", "counts": split_counts,
                   "group_counts": split_counts, "intersections": {
                       "train_validation": len(main_groups_by_split["train"] & main_groups_by_split["validation"]),
                       "train_test": len(main_groups_by_split["train"] & main_groups_by_split["test"]),
                       "validation_test": len(main_groups_by_split["validation"] & main_groups_by_split["test"])}}
    q_splits = {split: set(_q_hashes(main.arrays["q"][main.arrays["split"] == split.encode()])) for split in ("train", "validation", "test")}
    duplicate_audit = {"status": "PASS", "duplicate_sample_ids": 0, "duplicate_group_ids_in_independent_roots": 0,
                       "cross_split_q_duplicates": {"train_validation": len(q_splits["train"] & q_splits["validation"]),
                                                    "train_test": len(q_splits["train"] & q_splits["test"]),
                                                    "validation_test": len(q_splits["validation"] & q_splits["test"])},
                       "hard_subset_exact_overlaps": overlap}
    fk_summary = {"status": "PASS", "record_count": len(main_q)+len(boundary_q)+len(singular_q),
                  "maximum_position_error_m": max(x.max_position_error for x in (main,boundary,singular)),
                  "maximum_rotation_frobenius_error": max(x.max_rotation_error for x in (main,boundary,singular)),
                  "position_threshold_m": position_limit, "rotation_threshold": rotation_limit,
                  "nonfinite_count": 0, "joint_limit_violation_count": 0, "quaternion_reconstruction_failures": 0}
    distances = np.minimum((boundary_q-bounds[:,0])/(bounds[:,1]-bounds[:,0]), (bounds[:,1]-boundary_q)/(bounds[:,1]-bounds[:,0]))
    hard_summary = {"status": "PASS", "boundary": {"count": len(boundary_q), "threshold_strict_lt": config["hard_subsets"]["boundary"]["normalized_joint_limit_distance_lt"],
                    "maximum_selected_min_distance": float(np.max(np.min(distances, axis=1))),
                    "joint_side_counts": dict(Counter(f"joint_{x['joint_index']+1}_{x['side']}" for x in boundary_labels)),
                    "meaning": "joint-limit boundary, not Cartesian workspace boundary"},
                    "singularity": {"count": len(singular_q), "threshold": threshold, "threshold_source": "main train split only",
                    "candidate_count": scanned, "acceptance_rate": len(singular_q)/scanned}, "overlap": overlap}
    manifest = {"schema_version": "1.0.0", "status": "PASS", "generation_path": output.as_posix(),
                "reproduction_command": f"pixi run --locked python -m neurokinematics.data.cli generate --output {output.as_posix()}",
                "not_tracked_reason": "Full deterministic shards are generated artifacts and are ignored by Git.",
                "record_counts": {"main": len(main_q), "boundary": len(boundary_q), "singularity": len(singular_q)},
                "config_sha256": canonical_json_hash(config_path), "schema_sha256": canonical_json_hash(schema_path),
                "robot_hashes": inputs.hashes, "shards": shards, "dataset_content_sha256": _dataset_hash(shards)}
    _write_json(output/"dataset-manifest.json", manifest)
    if evidence:
        evidence = Path(evidence); evidence.mkdir(parents=True, exist_ok=True)
        items = {"dataset-manifest.json": manifest, "split-audit.json": split_audit,
                 "duplicate-audit.json": duplicate_audit, "fk-validation-summary.json": fk_summary,
                 "hard-subsets-summary.json": hard_summary, "coverage-summary.json": coverage_summary,
                 "coverage-sensitivity.json": sensitivity, "normalization.json": normalization(main.arrays),
                 "sample-hashes.json": {"main_q_sha256": hashlib.sha256(main_q.tobytes()).hexdigest(),
                                        "boundary_q_sha256": hashlib.sha256(boundary_q.tobytes()).hexdigest(),
                                        "singularity_q_sha256": hashlib.sha256(singular_q.tobytes()).hexdigest(),
                                        "first_five_records": [{"sample_id": main.arrays["sample_id"][i].decode(), "q": main_q[i].tolist(), "position_m": main.arrays["position_m"][i].tolist(), "quaternion_wxyz": main.arrays["quaternion_wxyz"][i].tolist()} for i in range(5)]},
                 "environment.json": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "pinocchio": __import__("pinocchio").__version__, "linux_execution": "NOT_RUN"}}
        for name, value in items.items(): _write_json(evidence/name, value)
    return manifest


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def verify_dataset(output: Path, manifest_path: Path, config_path: Path = CONFIG_PATH, schema_path: Path = SCHEMA_PATH) -> dict:
    output, manifest_path = Path(output), Path(manifest_path)
    manifest, schema, config = _json(manifest_path), _json(schema_path), _json(config_path)
    if manifest["config_sha256"] != sha256_file(config_path) or manifest["schema_sha256"] != sha256_file(schema_path):
        raise ValueError("manifest/config/schema mismatch")
    if config["immutable_inputs"] != {"urdf_sha256": load_robot().hashes["assets/robots/robot_a/robot.urdf"],
        "robot_spec_sha256": load_robot().hashes["assets/robots/robot_a/robot_spec.json"],
        "robot_manifest_sha256": load_robot().hashes["assets/robots/robot_a/manifest.json"],
        "tcp_sha256": load_robot().hashes["config/robots/tcp_tool0.json"],
        "f0_02_sample_sha256": config["immutable_inputs"]["f0_02_sample_sha256"], "f0_03_sample_sha256": config["immutable_inputs"]["f0_03_sample_sha256"]}:
        raise ValueError("immutable input mismatch")
    field_order = [field["name"] for field in schema["fields"]]
    inputs = load_robot()
    expected_metadata = {"robot_id": inputs.robot_id,
                         "model_hash": inputs.hashes["assets/robots/robot_a/robot.urdf"],
                         "tcp_hash": inputs.hashes["config/robots/tcp_tool0.json"]}
    verified = 0
    for subset in ("main", "boundary", "singularity"):
        for expected in manifest["shards"][subset]:
            path = output/expected["path"]
            if not path.is_file() or sha256_file(path) != expected["file_sha256"]:
                raise ValueError("missing or corrupted shard")
            arrays = read_shard(path, field_order)
            validate_arrays(arrays, schema, inputs.limits, expected_metadata)
            audit_groups_and_duplicates(arrays)
            if canonical_array_hash(arrays, field_order) != expected["content_sha256"]:
                raise ValueError("shard content hash mismatch")
            verified += 1
    if _dataset_hash(manifest["shards"]) != manifest["dataset_content_sha256"]:
        raise ValueError("dataset content hash mismatch")
    return {"status": "PASS", "verified_shards": verified, "dataset_content_sha256": manifest["dataset_content_sha256"]}
