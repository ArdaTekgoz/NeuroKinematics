"""Build the additive recovery decision manifest from completed evidence."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"experiments/C1-06S"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def entry(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


freeze, final = read(BASE/"final-freeze.json"), read(BASE/"final1.json")
for name, digest in freeze["files"].items():
    if entry(ROOT/name)["sha256"] != digest:
        raise ValueError("source drift "+name)
for file in ("validation-audit.json", "final1-audit.json", "clean-replay.json"):
    if read(BASE/file)["status"] != "PASS":
        raise ValueError("missing gate "+file)
if final["status"] != "PASS" or final["groups"]["main"]["profile_a_rate"] < .95 or final["false_acceptance"]:
    raise ValueError("product gate")
if final["unique_queries"] != 12000 or final["measurements"] != 60000:
    raise ValueError("denominator")
tests = 0
for name in ("unit-r3.xml", "f001.xml", "f002.xml", "f003.xml"):
    suites = list(ET.parse(BASE/name).getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")):
        raise ValueError("test gate "+name)
    tests += sum(int(s.attrib["tests"]) for s in suites)
if tests != 315:
    raise ValueError("test count")
old_handoff = read(ROOT/"experiments/C1-07/HYBRID_HANDOFF.json")
handoff = dict(schema="core-geometric-handoff-v1", component_id="analytic-kr6-r1",
    status="READY_FOR_H2_01", task="C1-06S", learned=False, robot=old_handoff["robot"],
    source=entry(ROOT/"src/neurokinematics/solvers/analytic.py"),
    frozen_manifest=entry(BASE/"final-freeze.json"), component_card=entry(BASE/"COMPONENT_CARD.md"),
    architecture_decision=entry(ROOT/"docs/adr/ADR-026-geometric-core-recovery.md"),
    api="AnalyticIK.solve(q_current, target_position_m, target_quaternion_wxyz)",
    historical_neural_handoff=entry(ROOT/"experiments/C1-07/HYBRID_HANDOFF.json"),
    geometric_gate="PASS", direct_neural_gate="NO_GO_UNCHANGED", old_h2="REJECTED_UNCHANGED",
    h1="NOT_MEASURED", hybrid_service="NOT_STARTED", collision="NOT_CHECKED",
    next_task="H2-01", deadline_service="NOT_IMPLEMENTED", artifact_storage="LOCAL_ONLY",
    metrics=final["groups"]["all"], tests_passed=tests,
    fresh_environment="NOT_CREATED", clean_source_replay="12000_EXACT")
write(BASE/"HANDOFF.json", handoff)
write(BASE/"acceptance.json", dict(task="C1-06S", status="PASS / ACCEPTED", scope="ROBOT_SPECIFIC_GEOMETRIC_IK",
    tests={"T-C05S":"PASS", "T-C06S":"PASS_EXISTING_LOCKED_RUNTIME"}, tests_passed=tests,
    main_profile_a=final["groups"]["main"]["profile_a"], main_n=10000,
    all_profile_a=final["groups"]["all"]["profile_a"], all_profile_b=final["groups"]["all"]["profile_b"],
    all_n=final["unique_queries"], false_acceptance=final["false_acceptance"],
    requirements=["REQ-C02", "REQ-C03", "REQ-C05", "REQ-C06"],
    original_neural_decision="UNCHANGED", next_task="H2-01", h1="NOT_MEASURED",
    evidence=[entry(BASE/name) for name in ("validation.json", "validation-audit.json", "final-freeze.json",
        "final1-inputs.json", "final1.json", "final1-audit.json", "clean-replay.json", "HANDOFF.json")]))
paths = list(BASE.rglob("*"))
paths += [ROOT/name for name in ("src/neurokinematics/solvers/analytic.py", "tests/c1_06s/test_analytic.py",
    "scripts/evaluate_c106s.py", "scripts/test_c106s.py", "scripts/reproduce_c106s.py", "scripts/replay_c106s.py",
    "scripts/finalize_c106s.py", "docs/tasks/C1-06S.md", "docs/adr/ADR-026-geometric-core-recovery.md")]
for folder in (ROOT/"experiments/C1-06R/commands").iterdir():
    if folder.name[:3].isdigit() and 62 <= int(folder.name[:3]) <= 71:
        paths += list(folder.rglob("*"))
write(BASE/"evidence-manifest.json", dict(status="PASS", files=[entry(p) for p in sorted(set(paths)) if p.is_file()]))
print("C1-06S PASS; 315 tests; 12000 independent targets; 60000 audited measurements; H2-01 ready")
