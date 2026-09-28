"""Read-only C1-02 contract and frozen-input preflight; no pair generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/C1-02"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_hash(path: str, expected: str) -> str:
    actual = digest(ROOT / path) if (ROOT / path).is_file() else "MISSING"
    if actual != expected:
        raise ValueError(f"SHA_MISMATCH path={path} expected={expected} actual={actual}")
    return actual


def validate_contract(config: dict, schema: dict) -> None:
    source = config["source"]
    assert sum(source["subsets"].values()) == 12000
    assert source["planned_rows"] == 2 * sum(source["subsets"].values())
    assert source["planned_mode_counts"] == {"local": 12000, "wide": 12000}
    assert {mode: sum(counts[mode] for counts in source["planned_split_mode_counts"].values())
            for mode in ("local", "wide")} == source["planned_mode_counts"]
    assert all(counts["local"] == counts["wide"] for counts in source["planned_split_mode_counts"].values())
    assert config["teacher"]["candidate_budget_per_wide_row"] == len(config["teacher"]["candidate_starts"])
    assert config["pilot"]["max_solver_calls"] == config["pilot"]["wide_rows"] * len(config["teacher"]["candidate_starts"])
    assert config["pilot"]["max_solver_iterations"] == config["pilot"]["max_solver_calls"] * config["teacher"]["max_iterations_per_call"]
    assert config["normalization"]["input_fields"] == ["position_m", "quaternion_wxyz", "q_current"]
    assert not set(config["normalization"]["input_fields"]) & set(config["normalization"]["forbidden_input_fields"])
    fields = schema["fields"]
    names = [field["name"] for field in fields]
    assert len(names) == len(set(names))
    assert {"group_id", "source_sample_id", "source_manifest_sha256", "pair_mode", "q_current", "q_target", "label_present", "teacher_status", "teacher_failure_class"} <= set(names)
    assert set(field["name"] for field in fields if field.get("role") == "model_input") == set(config["normalization"]["input_fields"])
    assert next(field for field in fields if field["name"] == "q_target")["role"] == "label_only"


def expected_inputs() -> dict[str, str]:
    handoff = read_json(ROOT / "experiments/F0-06/handoff-inputs.json")
    c101 = read_json(ROOT / "experiments/C1-01/frozen-hashes.json")["files"]
    manifest = read_json(ROOT / "experiments/F0-04/dataset-manifest.json")
    paths = dict(handoff)
    for path, value in c101.items():
        if path in paths and paths[path] != value:
            raise ValueError(f"upstream expected SHA conflict: {path}")
        paths[path] = value
    source_root = Path("data/generated/F0-04/run-a")
    for subset in ("main", "boundary", "singularity"):
        for shard in manifest["shards"][subset]:
            paths[(source_root / shard["path"]).as_posix()] = shard["file_sha256"]
    paths["data/generated/F0-05/acceptance/run-a/query-list.jsonl"] = read_json(ROOT / "experiments/F0-05/query-manifest.json")["query_list_sha256"]
    paths["experiments/C1-01/udp-v2/full/gate.json"] = "d7b9126583429f93c8629708b11633b99ffaf87ca664799c4bb01665924fb83d"
    paths["experiments/C1-01/udp-v2/verify/gate.json"] = "eda4bf6f815790aaebba4146c5369740e3a1d70d1fc8429cfd6f1084269c8d0d"
    verify_gate = read_json(ROOT / "experiments/C1-01/udp-v2/verify/gate.json")
    if verify_gate["status"] != "PASS" or verify_gate["full_gate_sha256"] != paths["experiments/C1-01/udp-v2/full/gate.json"]:
        raise ValueError("C1-01 verify gate status or full binding mismatch")
    for name, expected in verify_gate["files"].items():
        paths[f"experiments/C1-01/udp-v2/verify/{name}"] = expected
    for path in (".gitattributes", "src/neurokinematics/solvers/dls.py", "experiments/C1-01/udp-v2/runtime-lock.json", "experiments/C1-01/udp-v2-verify-evidence-check.json", "experiments/C1-02/config.json", "experiments/C1-02/schema.json", "scripts/check_c102_stage1.py"):
        paths[path] = digest(ROOT / path)
    return dict(sorted(paths.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if sum((args.write_manifest, args.check, args.self_test)) != 1:
        parser.error("choose one mode")
    config = read_json(EVIDENCE / "config.json")
    schema = read_json(EVIDENCE / "schema.json")
    validate_contract(config, schema)
    if args.self_test:
        tests = 0
        def must_fail(call):
            nonlocal tests
            try:
                call()
            except (AssertionError, ValueError):
                tests += 1
                return
            raise AssertionError("mutation survived")
        bad = json.loads(json.dumps(config))
        bad["normalization"]["input_fields"].append("q_target")
        must_fail(lambda: validate_contract(bad, schema))
        bad = json.loads(json.dumps(config))
        bad["source"]["planned_split_mode_counts"]["test"]["wide"] -= 1
        must_fail(lambda: validate_contract(bad, schema))
        bad_schema = json.loads(json.dumps(schema))
        next(f for f in bad_schema["fields"] if f["name"] == "q_target")["role"] = "model_input"
        must_fail(lambda: validate_contract(config, bad_schema))
        must_fail(lambda: require_hash("experiments/C1-02/config.json", "0" * 64))
        print(json.dumps({"stage1_contract": "PASS", "mutations_caught": tests, "production_rows": "NOT_RUN", "T-C07": "NOT_RUN"}))
        return
    paths = expected_inputs()
    for path, expected in paths.items():
        require_hash(path, expected)
    output = EVIDENCE / "input-hashes.json"
    if args.write_manifest:
        output.write_text(json.dumps({"schema_version": "1.0.0", "status": "STAGE_1_VERIFIED", "files": paths}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        frozen = read_json(output)["files"]
        if frozen != paths:
            raise ValueError("stage1 frozen manifest differs from current inputs")
    print(json.dumps({"status": "PASS", "file_count": len(paths), "source_shards": 12, "production_rows": "NOT_RUN", "T-C07": "NOT_RUN"}))


if __name__ == "__main__":
    main()
