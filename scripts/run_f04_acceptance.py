"""Frozen F0-04 acceptance runner; stops at the first real failure."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "85897aad33a5c5c91442859bc6ea0be2809ee1ec"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def verify_evidence(output):
    lines = (output/"SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not lines: raise ValueError("empty SHA256SUMS")
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f"evidence hash mismatch: {relative}")
    return len(lines)


def write_hashes(output):
    files = [p for p in output.iterdir() if p.is_file() and p.suffix != ".md" and p.name != "SHA256SUMS"]
    for folder in ("src/neurokinematics/data", "tests/f0_04"):
        files.extend((ROOT/folder).glob("*.py"))
    files.extend(ROOT/p for p in ("pixi.toml","pixi.lock","scripts/run_f04_acceptance.py",
                                  "experiments/F0-04/config.json","experiments/F0-04/schema.json"))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in sorted(set(files))]
    (output/"SHA256SUMS").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")


def preflight():
    from neurokinematics.kinematics.model import load_robot
    from neurokinematics.kinematics.validation import sample_configurations as f02_samples, sample_hash
    from neurokinematics.kinematics.jacobian_validation import sample_configurations as f03_samples
    branch = subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    remote = subprocess.check_output(["git","rev-parse","origin/main"],cwd=ROOT,text=True).strip()
    if branch != "main" or head != EXPECTED_HEAD or remote != EXPECTED_HEAD:
        raise ValueError(f"unexpected Git start: {branch=} {head=} {remote=}")
    for commit in ("981f6143ce38574021edac7373586976cf97bdf4", EXPECTED_HEAD):
        subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=ROOT,check=True)
    inputs=load_robot()
    f02=sample_hash(f02_samples(inputs)); f03=sample_hash(f03_samples(inputs))
    if f02 != "8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101": raise ValueError("F0-02 sample hash mismatch")
    if f03 != "678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7": raise ValueError("F0-03 sample hash mismatch")
    return {"status":"PASS","branch":branch,"head":head,"origin_main":remote,"input_hashes":inputs.hashes,
            "f0_02_sample_sha256":f02,"f0_03_sample_sha256":f03}


def junit_count(path):
    cases=list(ET.parse(path).getroot().iter("testcase"))
    bad=[c.get("name") for c in cases if any(c.find(tag) is not None for tag in ("failure","error","skipped"))]
    if not cases or bad: raise ValueError(f"incomplete JUnit: {path.name}: {bad}")
    return len(cases)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("experiments/F0-04"))
    parser.add_argument("--generated-root",type=Path,default=Path("data/generated/F0-04"))
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args(); evidence=(ROOT/args.output).resolve(); generated=(ROOT/args.generated_root).resolve()
    evidence.relative_to(ROOT); generated.relative_to(ROOT)
    if args.verify_only:
        print(f"SHA256SUMS: {verify_evidence(evidence)} files verified"); return 0
    evidence.mkdir(parents=True,exist_ok=True)
    write_json(evidence/"preflight.json",preflight())
    run_a,run_b=generated/"run-a",generated/"run-b"
    for target in (run_a,run_b):
        if target.exists():
            target.resolve().relative_to(generated)
            shutil.rmtree(target)
    def junit(name): return f"--junitxml={(evidence/name).relative_to(ROOT).as_posix()}"
    commands=[
      ["pixi","lock","--check"], ["pixi","install","--locked"],
      ["pixi","run","--locked","test-f00",junit("f00-junit.xml")],
      ["pixi","run","--locked","test-f01",junit("f01-junit.xml")],
      ["pixi","run","--locked","test-f02",junit("f02-junit.xml"),"-o","junit_family=legacy"],
      ["pixi","run","--locked","test-f03",junit("f03-junit.xml"),"-o","junit_family=legacy"],
      ["pixi","run","--locked","python","-m","pytest","-q","tests/f0_04/test_unit.py",junit("f04-unit-junit.xml")],
      ["pixi","run","--locked","python","-m","pytest","-q","tests/f0_04/test_tf05_determinism.py",junit("tf05-junit.xml")],
      ["pixi","run","--locked","python","-m","pytest","-q","tests/f0_04/test_tf06_split.py",junit("tf06-junit.xml")],
      ["pixi","run","--locked","python","-m","pytest","-q","tests/f0_04/test_tf07_accuracy_coverage.py",junit("tf07-junit.xml")],
      ["pixi","run","--locked","python","-m","pytest","-q","tests/f0_04/test_mutations.py",junit("mutation-junit.xml")],
      ["pixi","run","--locked","python","-m","neurokinematics.data.cli","generate","--output",str(run_a.relative_to(ROOT)),"--evidence",str(evidence.relative_to(ROOT))],
      ["pixi","run","--locked","python","-m","neurokinematics.data.cli","generate","--output",str(run_b.relative_to(ROOT))],
      ["pixi","run","--locked","python","-m","neurokinematics.data.cli","verify","--output",str(run_a.relative_to(ROOT)),"--manifest",str((evidence/"dataset-manifest.json").relative_to(ROOT))],
    ]
    env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1"}; records=[]
    for command in commands:
        started=datetime.now(timezone.utc).isoformat()
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",env=env)
        records.append({"command":command,"exit_code":result.returncode,"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"stdout":result.stdout,"stderr":result.stderr})
        write_json(evidence/"commands.json",records)
        print("$ "+" ".join(command),flush=True); print(result.stdout+result.stderr,flush=True)
        if result.returncode: write_json(evidence/"acceptance.json",{"status":"FAIL","failed_command":command}); return result.returncode
    manifest_a=json.loads((evidence/"dataset-manifest.json").read_text(encoding="utf-8"))
    manifest_b=json.loads((run_b/"dataset-manifest.json").read_text(encoding="utf-8"))
    pairs=[]
    for subset in ("main","boundary","singularity"):
        for a,b in zip(manifest_a["shards"][subset],manifest_b["shards"][subset],strict=True):
            pairs.append({"subset":subset,"index":a["index"],"file_match":a["file_sha256"]==b["file_sha256"],"content_match":a["content_sha256"]==b["content_sha256"]})
    deterministic={"status":"PASS" if manifest_a["dataset_content_sha256"]==manifest_b["dataset_content_sha256"] and all(x["file_match"] and x["content_match"] for x in pairs) else "FAIL",
                   "dataset_content_match":manifest_a["dataset_content_sha256"]==manifest_b["dataset_content_sha256"],"shards":pairs}
    write_json(evidence/"determinism-summary.json",deterministic)
    if deterministic["status"] != "PASS": raise ValueError("independent productions differ")
    names=("f00","f01","f02","f03","f04-unit","tf05","tf06","tf07","mutation")
    counts={name:junit_count(evidence/f"{name}-junit.xml") for name in names}
    cases=ET.parse(evidence/"mutation-junit.xml").getroot().iter("testcase"); mutations=[]
    for case in cases:
        props={p.get("name"):p.get("value") for p in case.findall("properties/property")}
        if "mutation" in props: mutations.append(props)
    if len(mutations)!=17 or any(x["detected"]!="True" for x in mutations): raise ValueError("mutation evidence incomplete")
    write_json(evidence/"mutation-results.json",{"status":"PASS","detected_count":len(mutations),"cases":mutations})
    write_json(evidence/"acceptance.json",{"status":"PASS","test_counts":counts,"command_count":len(records),
               "dataset_content_sha256":manifest_a["dataset_content_sha256"],"finished_utc":datetime.now(timezone.utc).isoformat()})
    write_hashes(evidence); verified=verify_evidence(evidence)
    write_json(evidence/"evidence-verification.json",{"status":"PASS","verified_file_count":verified})
    # Include the final verification record itself, then freeze and verify once more.
    write_hashes(evidence); verified=verify_evidence(evidence)
    print(f"F0-04 PASS; SHA256SUMS: {verified} files verified")
    return 0


if __name__=="__main__": raise SystemExit(main())
