"""Stage 1 frozen artifacts and strict result-contract validation.

The structural validator implements only the explicitly used JSON Schema
keywords below, rejecting unsupported keywords. It is not a general-purpose
JSON Schema implementation. Semantic and independent FK checks follow it.
No query generation, JSONL IO, timing runner or CLI is provided here.
"""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from neurokinematics.kinematics.metrics import quaternion_wxyz
from neurokinematics.kinematics.model import ROOT, validate_q
from neurokinematics.solvers.dls import SolverResult, SolverStatus
from .validation import CandidateValidator, deadline_success

def strict_json(text):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    finite_json(value)
    return value


def finite_json(value):
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise ValueError("JSON keys must be strings")
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("unsupported JSON value")


def validate_structure(value, schema, path="result"):
    supported = {"$schema", "title", "description", "type", "const", "enum", "anyOf",
                 "properties", "required", "additionalProperties", "items", "minItems",
                 "maxItems", "minimum", "maximum", "minLength", "pattern"}
    if set(schema) - supported:
        raise ValueError("unsupported schema keyword")
    finite_json(value)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                validate_structure(value, branch, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: no schema branch matches")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    if kind == "number":
        ok = type(value) in (int, float)
    else:
        ok = kind is None or type(value) is types[kind]
    if not ok:
        raise ValueError(f"{path}: invalid type")
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        raise ValueError(f"{path}: invalid constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum")
    if kind == "object":
        if set(schema["required"]) - value.keys():
            raise ValueError(f"{path}: missing field")
        if schema.get("additionalProperties") is False and value.keys() - schema["properties"].keys():
            raise ValueError(f"{path}: unexpected field")
        for key, item in value.items():
            validate_structure(item, schema["properties"][key], f"{path}.{key}")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: wrong length")
        for index, item in enumerate(value):
            validate_structure(item, schema["items"], f"{path}[{index}]")
    if kind in ("number", "integer"):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{path}: out of range")
    if kind == "string":
        if len(value) < schema.get("minLength", 0) or ("pattern" in schema and not re.fullmatch(schema["pattern"], value)):
            raise ValueError(f"{path}: invalid string")


def load_frozen(root=ROOT):
    directory = Path(root) / "experiments/F0-05"
    lock = strict_json((directory / "stage1-frozen-hashes.json").read_text(encoding="utf-8"))
    result = {}
    for relative, expected in lock["artifacts"].items():
        raw = (Path(root) / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"frozen artifact hash mismatch: {relative}")
        result[Path(relative).name] = strict_json(raw.decode("utf-8"))
    return result


def load_reproduction_config(path):
    """Load a smaller benchmark while preserving every frozen numeric rule.

    Only query counts, measurement passes and warm-up count may change.
    The caller supplies the separately generated dataset manifest explicitly.
    """
    frozen = load_frozen()['config.json']
    config = strict_json(Path(path).read_text(encoding='utf-8'))
    allowed = {'query_counts', 'measurement_passes', 'warmup_queries_per_deadline'}
    if set(config) != set(frozen) or any(config[k] != frozen[k] for k in frozen if k not in allowed):
        raise ValueError('reproduction changed a frozen benchmark rule')
    if set(config['query_counts']) != set(frozen['query_counts']):
        raise ValueError('reproduction subset mismatch')
    for key, count in config['query_counts'].items():
        if type(count) is not int or count < 2 or count % 2 or count > frozen['query_counts'][key]:
            raise ValueError('reproduction counts must be positive even subsets of production size')
    for key in ('measurement_passes', 'warmup_queries_per_deadline'):
        if type(config[key]) is not int or not 1 <= config[key] <= frozen[key]:
            raise ValueError('invalid reproduction pass/warm-up count')
    return config


def validate_record(record, *, validator: CandidateValidator, expected_hashes):
    """Validate one in-memory record and recompute its independent FK verdict.

    expected_hashes must come from separately verified manifests, not the row.
    Proven-unreachable records require a future certificate-aware entry point;
    this Stage 1 function refuses that claim rather than trusting a proof label.
    """
    artifacts = load_frozen()
    validate_structure(record, artifacts["benchmark-schema.json"])
    if (record['frame'] != validator.inputs.base or record['tcp'] != validator.inputs.tcp
            or record['joint_order'] != list(validator.inputs.joint_names)):
        raise ValueError("validator frame/TCP/joint contract mismatch")
    inputs = artifacts['config.json']['immutable_file_sha256']
    if hashlib.sha256(validator.inputs.urdf).hexdigest() != inputs['assets/robots/robot_a/robot.urdf']:
        raise ValueError("validator model identity mismatch")
    expected_keys = {"query_list_sha256", "solver_config_sha256", "dataset_manifest_sha256"}
    if set(expected_hashes) != expected_keys:
        raise ValueError("all three independent expected hashes required")
    if any(record[k] != v for k, v in expected_hashes.items()):
        raise ValueError("record hash binding mismatch")
    status = SolverStatus(record["solver_status"])
    if "PROVEN_UNREACHABLE" in (status, record["validation_status"], record["reachability"]):
        raise ValueError("independent analytic certificate verification required")
    if record["reachability_proof_sha256"] is not None:
        raise ValueError("unexpected proof without proven reachability")
    if record["total_elapsed_ns"] != record["solve_elapsed_ns"] + record["validation_elapsed_ns"]:
        raise ValueError("elapsed interval mismatch")
    iterations = record["iterations"]
    if (iterations is None) != (record["iteration_availability"] == "NOT_AVAILABLE"):
        raise ValueError("iteration availability mismatch")
    if status == SolverStatus.INVALID_INPUT:
        if iterations is not None or record["input_error"] is None:
            raise ValueError("invalid input must preserve unavailable iteration and diagnosis")
        if record['q_candidate'] is not None:
            raise ValueError("invalid DLS input cannot produce a candidate")
    elif iterations is None or record["input_error"] is not None:
        raise ValueError("valid DLS input requires measured iterations")
    hits = [record["first_profile_a_iteration"], record["first_profile_b_iteration"]]
    if any(hit is not None and (iterations is None or hit > iterations) for hit in hits):
        raise ValueError("first hit exceeds completed iterations")
    if hits[1] is not None and (hits[0] is None or hits[0] > hits[1]):
        raise ValueError("Profile B hit requires earlier or simultaneous A hit")
    if status == SolverStatus.SUCCESS and hits[1] is None:
        raise ValueError("DLS success requires internal Profile B hit")
    if status == SolverStatus.MAX_ITERATIONS and iterations != 200:
        raise ValueError("max iterations must reflect the frozen cap")
    if record["target_source"] == "independent_frozen_fk":
        if record["subset"] not in ("main", "boundary", "singularity") or record["start_class"] not in ("local", "wide"):
            raise ValueError("invalid benchmark query class")
        if record["reachability"] != "KNOWN_REACHABLE" or record["thermal_state"] != "warm":
            raise ValueError("frozen FK queries are kinematically reachable warm measurements")
    if status != SolverStatus.INVALID_INPUT:
        validate_q(record["q_current"], validator.inputs.joint_names, validator.inputs.limits)
        quaternion_wxyz(record["target_quaternion_wxyz"])
    candidate = record["q_candidate"]
    result = SolverResult(status, record["termination_reason"],
                          None if candidate is None else np.asarray(candidate, dtype=np.float64),
                          iterations, *hits, record["solve_elapsed_ns"])
    verdict = validator.validate(result, record["target_position_m"], record["target_quaternion_wxyz"])
    for name in ("position_error_m", "orientation_error_rad", "orientation_error_deg",
                 "profile_a_geometry", "profile_b_geometry", "joint_limits", "collision"):
        expected, actual = getattr(verdict, name), record[name]
        if type(expected) is float:
            matches = actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"independent verdict mismatch: {name}")
    if record["validation_status"] != verdict.status:
        raise ValueError("independent validation status mismatch")
    for profile in ("a", "b"):
        expected = deadline_success(record[f"profile_{profile}_geometry"], status,
                                    record["total_elapsed_ns"], record["deadline_profile_ms"] * 1_000_000)
        if record[f"profile_{profile}_deadline"] != expected:
            raise ValueError("deadline verdict mismatch")
