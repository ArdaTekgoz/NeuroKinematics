"""Read-only C1-01 Stage 1 gate, frozen contract, and mutation checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/C1-01/baseline-config.json"
MANIFEST = ROOT / "experiments/C1-01/frozen-hashes.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate(config: dict, handoff: dict, query_manifest: dict) -> None:
    require(config["schema_version"] == "1.0.0", "config schema version")
    require(config["stage"] == "STAGE_1_CONTRACT_ONLY", "stage status")
    require((config["requirement"], config["acceptance_test"]) == ("REQ-C01", "T-C00"), "traceability")
    execution = config["execution"]
    for key in ("os", "architecture", "ros_distro", "moveit_version", "moveit_source_commit", "worker", "timing", "cpu_policy"):
        require(isinstance(execution[key], str) and bool(execution[key]), f"execution.{key}")
    require(execution["os"] == "Ubuntu 24.04 LTS" and execution["architecture"] == "x86_64" and execution["ros_distro"] == "jazzy", "platform")
    require(execution["max_benchmark_processes"] == 1 and execution["max_solver_threads"] == 2 and execution["cpu_affinity_logical_count"] == 2, "resources")
    robot = config["robot"]
    require(robot["id"] == "kuka_kr6_r900_sixx", "robot identity")
    require((robot["base_frame"], robot["tcp_frame"], robot["position_unit"], robot["joint_unit"], robot["quaternion_order"]) == ("base_link", "tool0", "m", "rad", "wxyz"), "robot frame/units")
    require(robot["collision"] == "NOT_CHECKED", "collision")
    require(robot["joint_order"] == [f"joint_{i}" for i in range(1, 7)], "joint order")
    for key, path in (("urdf_sha256", "assets/robots/robot_a/robot.urdf"), ("robot_spec_sha256", "assets/robots/robot_a/robot_spec.json"), ("tcp_sha256", "config/robots/tcp_tool0.json")):
        require(robot[key] == handoff[path], f"robot hash {key}")
    queries = config["queries"]
    require(queries["manifest_sha256"] == handoff[queries["manifest_path"]], "query manifest hash")
    require(queries["list_sha256"] == query_manifest["query_list_sha256"], "query list hash")
    require(queries["count"] == query_manifest["record_count"] == 12000, "query count")
    require(queries["subsets"] == {k: query_manifest["subsets"][k]["accepted"] for k in ("main", "boundary", "singularity")}, "query subsets")
    require("never sent" in queries["q_target_policy"], "q_target isolation")
    benchmark = config["benchmark"]
    frozen = json.loads((ROOT / "experiments/F0-05/config.json").read_text(encoding="utf-8"))
    require(benchmark["profiles"] == frozen["profiles"], "profile tolerance")
    require(benchmark["deadline_profiles_ms"] == frozen["deadline_profiles_ms"], "deadline profiles")
    require(benchmark["measurement_passes"] == frozen["measurement_passes"] == 5, "measurement passes")
    require(benchmark["per_solver_expected_attempts"] == queries["count"] * 2 * 5, "attempt count")
    require(benchmark["iteration_missing"] == "NOT_AVAILABLE", "iteration missing")
    for key in ("validation", "deadline_rule", "timeout_policy", "raw_results"):
        require(bool(benchmark[key]), f"benchmark.{key}")
    adapter = config["adapter"]
    for key in ("boundary", "pose_conversion", "joint_mapping", "candidate_validation", "result_schema"):
        require(bool(adapter[key]), f"adapter.{key}")
    for key in ("query_id", "q_current", "target_position_m", "target_quaternion_wxyz", "deadline_ns", "solver_config_sha256"):
        require(key in adapter["inputs"], f"adapter input {key}")
    for key in ("native_status", "common_status", "q_candidate_or_null", "total_elapsed_ns", "collision", "error_class"):
        require(key in adapter["outputs"], f"adapter output {key}")
    solvers = config["solvers"]
    ids = [s["id"] for s in solvers]
    require(len(solvers) == len(set(ids)) == 5, "unique solver identities")
    require(set(ids) == {"dls/default", "kdl/default", "trac_ik/speed", "pick_ik/local", "pick_ik/global"}, "required solver variants")
    for solver in solvers:
        for key in ("id", "name", "variant", "upstream_url", "version", "source_commit", "license", "maintenance", "integration", "parameters", "seed", "threads", "iterations", "internal_acceptance"):
            require(key in solver and solver[key] not in (None, "", {}), f"{solver.get('id')}.{key}")
        require(solver["upstream_url"].startswith("https://"), f"{solver['id']} source URL")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", solver["source_commit"])), f"{solver['id']} exact commit")
        require(solver["threads"] <= execution["max_solver_threads"], f"{solver['id']} threads")
    local = next(s for s in solvers if s["id"] == "pick_ik/local")
    global_ = next(s for s in solvers if s["id"] == "pick_ik/global")
    require(local["parameters"]["mode"] == "local" and global_["parameters"]["mode"] == "global", "pick_ik modes")
    require(local["parameters"] != global_["parameters"], "pick_ik config distinction")
    require("deprecated" in local["maintenance"] and "deprecated" in global_["maintenance"], "pick_ik maintenance")
    require(next(s for s in solvers if s["id"] == "dls/default")["parameters"]["config_sha256"] == handoff["experiments/F0-05/solver-config.json"], "DLS config hash")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results: list[dict] = []

    def check(name: str, action) -> None:
        try:
            action()
            results.append({"check": name, "status": "PASS"})
        except Exception as exc:
            results.append({"check": name, "status": "FAIL", "reason": str(exc)})

    handoff = json.loads((ROOT / "experiments/F0-06/handoff-inputs.json").read_text(encoding="utf-8"))
    for path, expected in handoff.items():
        check(f"handoff:{path}", lambda path=path, expected=expected: require(digest(ROOT / path) == expected, f"hash mismatch: {path}; expected {expected}; got {digest(ROOT / path) if (ROOT / path).is_file() else 'MISSING'}"))
    query_manifest = json.loads((ROOT / "experiments/F0-05/query-manifest.json").read_text(encoding="utf-8"))
    query_path = ROOT / query_manifest["file"]
    check("frozen query list", lambda: require(digest(query_path) == query_manifest["query_list_sha256"], "query list hash mismatch"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    check("config schema and binding", lambda: validate(config, handoff, query_manifest))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["files"].items():
        check(f"manifest:{path}", lambda path=path, expected=expected: require(HEX64.fullmatch(expected) is not None and digest(ROOT / path) == expected, f"manifest hash mismatch: {path}"))
    check("manifest query binding", lambda: require(manifest["query_list_sha256"] == query_manifest["query_list_sha256"], "manifest query binding"))
    check("G0 decision", lambda: require("PASS / ACCEPTED" in (ROOT / "experiments/F0-06/G0_DECISION.md").read_text(encoding="utf-8"), "G0 decision"))
    evidence_index = json.loads((ROOT / "experiments/F0-06/FOUNDATIONS_EVIDENCE_INDEX.json").read_text(encoding="utf-8"))
    check("F0-06 completion", lambda: require(evidence_index["tasks"]["F0-06"]["status"] == "COMPLETE", "F0-06 not complete"))

    mutations = [
        ("missing KDL", lambda c: c["solvers"].pop(1)),
        ("duplicate solver ID", lambda c: c["solvers"][1].update(id="dls/default")),
        ("merged pick modes", lambda c: c["solvers"][4]["parameters"].update(mode="local")),
        ("missing source URL", lambda c: c["solvers"][1].update(upstream_url="")),
        ("moving source ref", lambda c: c["solvers"][1].update(source_commit="jazzy")),
        ("missing license", lambda c: c["solvers"][2].update(license="")),
        ("wrong platform", lambda c: c["execution"].update(os="Windows 11")),
        ("wrong robot hash", lambda c: c["robot"].update(urdf_sha256="0" * 64)),
        ("wrong query hash", lambda c: c["queries"].update(list_sha256="0" * 64)),
        ("changed tolerance", lambda c: c["benchmark"]["profiles"]["A"].update(position_m=0.01)),
        ("missing timeout", lambda c: c["benchmark"].pop("timeout_policy")),
        ("changed deadline", lambda c: c["benchmark"].update(deadline_profiles_ms=[50])),
        ("changed passes", lambda c: c["benchmark"].update(measurement_passes=1)),
        ("changed joint order", lambda c: c["robot"]["joint_order"].reverse()),
        ("removed candidate validation", lambda c: c["adapter"].update(candidate_validation="")),
    ]
    for name, mutate in mutations:
        def negative(mutate=mutate):
            altered = copy.deepcopy(config)
            mutate(altered)
            try:
                validate(altered, handoff, query_manifest)
            except (ValueError, KeyError):
                return
            raise AssertionError("mutation escaped validation")
        check(f"mutation:{name}", negative)
    summary = {"task": "C1-01", "stage": "STAGE_1_CONTRACT_ONLY", "passed": sum(x["status"] == "PASS" for x in results), "failed": sum(x["status"] == "FAIL" for x in results), "checks": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("task", "stage", "passed", "failed")}, ensure_ascii=False))
    for item in results:
        if item["status"] == "FAIL":
            print(item)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
