"""Check C1-04 Stage 2 evidence and record the experimental decision."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/C1-04/stage2"
ACCEPTANCE = BASE / "acceptance.json"
IMPLEMENTATION_COMMIT = "7fde45052a5fe3ba3f31dd832541595b7c02313d"
CLEAN_NAMES = ("clean-pixi-install", "clean-venv-create", "clean-torch-lock-install",
               "clean-runtime-lock-install", "clean-pip-check", "clean-witness-check",
               "clean-commit", "clean-worktree-status")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command(name: str) -> tuple[dict, bytes]:
    item = read(BASE / "commands" / name / "command.json")
    for stream in ("stdout", "stderr"):
        raw = base64.b64decode(item[f"{stream}_base64"], validate=True)
        if digest(raw) != item[f"{stream}_sha256"]:
            raise ValueError(f"command raw {stream} SHA drift: {name}")
    return item, base64.b64decode(item["stdout_base64"], validate=True)


def collect() -> dict:
    commands = sorted((BASE / "commands").iterdir())
    checked = 0
    for folder in commands:
        if folder.is_dir():
            command(folder.name)
            checked += 1
    clean = {}
    for name in CLEAN_NAMES:
        item, stdout = command(name)
        if item["exit_code"] != 0:
            raise ValueError(f"clean command failed: {name}")
        clean[name] = {"start_utc": item["start_utc"], "end_utc": item["end_utc"],
                       "stdout_sha256": item["stdout_sha256"]}
    checkout, checkout_stdout = command("clean-commit")
    if checkout_stdout.decode("utf-8").strip() != IMPLEMENTATION_COMMIT:
        raise ValueError("clean checkout commit drift")
    if command("clean-worktree-status")[1].strip():
        raise ValueError("clean checkout had tracked or untracked files")
    witness = json.loads(command("clean-witness-check")[1].decode("utf-8"))
    if (witness.get("status") != "PASS" or witness.get("frozen_files") != 12
            or witness.get("checkpoints") != 6 or witness.get("samples_per_checkpoint") != 10
            or witness.get("max_q_abs_rad") != 0 or witness.get("max_fk_element_abs") != 0
            or witness.get("test_and_benchmark") != "SEALED_NOT_RUN"):
        raise ValueError("clean witness contract failed")
    audit = read(BASE / "audit.json")
    summary = read(BASE / "E-C01-summary.json")
    pilot = read(BASE / "pilot-summary.json")
    if (audit["status"] != "E_C01_EVIDENCE_COMPLETE" or audit["T-C03"] != "PASS"
            or audit["E-C01"] != "3_PAIRED_SEEDS_COMPLETE" or summary["status"] != "COMPLETE"
            or summary["split"] != "validation" or summary["test_and_benchmark"] != "SEALED_NOT_RUN"
            or pilot["t_c03_status"] != "PASS"):
        raise ValueError("pilot/E-C01 gate failed")
    if len(audit["results"]) != 6:
        raise ValueError("six checkpoint audits required")
    outcomes = []
    for seed_index, seed in enumerate((2026100201, 2026100202, 2026100203)):
        run = read(BASE / f"seed-{seed}-summary.json")
        paired = summary["seeds"][seed_index]
        if paired["seed"] != seed:
            raise ValueError("paired seed order drift")
        if run["epochs"] != 200 or run["optimizer_steps_per_model"] != 3000:
            raise ValueError("training budget mismatch")
        for variant in ("pose_only", "conditioned"):
            item = read(BASE / f"seed-{seed}-{variant}-validation.summary.json")
            evaluation = read(BASE / f"seed-{seed}-evaluation.json")["variants"][variant]
            overall = paired["models"][variant]["breakdowns"]["overall"]
            if (item["counts"]["rows"] != 3600 or overall["n"] != 3600
                    or overall["profile_a_success"] != 0
                    or item["rows_sha256"] != evaluation["rows_sha256"]):
                raise ValueError(f"validation outcome drift: {seed}/{variant}")
            row_path = BASE / f"seed-{seed}-{variant}-validation.jsonl"
            if digest(row_path.read_bytes()) != item["rows_sha256"]:
                raise ValueError(f"validation rows SHA drift: {seed}/{variant}")
            outcomes.append({"seed": seed, "variant": variant, "rows": 3600,
                             "profile_a_success": 0, "valid_raw": overall["valid_raw"],
                             "out_of_limits": overall["out_of_limits"],
                             "checkpoint_sha256": run["best_checkpoints"][variant]["sha256"]})
    return {"task": "C1-04", "status": "COMPLETE_EXPERIMENTAL_BASELINE",
            "requirement": "REQ-C03", "T-C03": "PASS", "E-C01": "THREE_PAIRED_SEEDS_COMPLETE",
            "direct_ik_decision": "NO_GO", "decision_reason": "Profile A 0/3600 for each of six validation runs",
            "next": "C1-05 E-C03 controlled supervised-plus-FK comparison",
            "next_status": "NOT_STARTED", "implementation_commit": IMPLEMENTATION_COMMIT,
            "clean_checkout_commit": checkout_stdout.decode("utf-8").strip(),
            "clean_witness": witness, "clean_commands": clean, "command_logs_verified": checked,
            "outcomes": outcomes, "weights": "LOCAL_ONLY", "remote_archive": "NOT_CONFIRMED",
            "test_and_benchmark": "SEALED_NOT_RUN", "physical_robot_safety": "NOT_CHECKED",
            "Linux": "NOT_RUN", "CUDA": "NOT_RUN"}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = collect()
    if args.write:
        ACCEPTANCE.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8", newline="\n")
    elif read(ACCEPTANCE) != actual:
        raise ValueError("C1-04 acceptance record drift")
    print(json.dumps({"status": "PASS", "decision": actual["direct_ik_decision"],
                      "runs": len(actual["outcomes"]), "commands": actual["command_logs_verified"]}))


if __name__ == "__main__":
    main()
