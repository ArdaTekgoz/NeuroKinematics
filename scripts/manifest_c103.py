"""Create or verify the C1-03 delivery manifest with explicit newline identity."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/C1-03/stage2"
MANIFEST = OUT / "evidence-manifest.json"
SUMS = OUT / "SHA256SUMS"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return data.replace(b"\r\n", b"\n")


def paths():
    selected = set((ROOT / "experiments/C1-03").rglob("*"))
    selected.update((ROOT / "tests/c1_03").glob("*.py"))
    selected.update(ROOT / name for name in (
        ".gitattributes", "src/neurokinematics/kinematics/torch_fk.py",
        "src/neurokinematics/core/torch_validation.py",
        "scripts/check_c103_stage1.py", "scripts/c103_command.py",
        "scripts/reproduce_c103.py", "scripts/audit_c103_artifacts.py",
        "scripts/finalize_c103.py", "scripts/manifest_c103.py",
        "docs/adr/ADR-011-c103-torch-fk.md",
        "docs/adr/ADR-012-c103-runtime-overlay.md", "docs/tasks/C1-03.md",
        "docs/records/STATUS.md", "docs/TRACEABILITY.md", "docs/roadmaps/C1_Core.md",
    ))
    return sorted(p for p in selected if p.is_file() and
                  p not in (MANIFEST, SUMS) and "__pycache__" not in p.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify-working-bytes", action="store_true",
                        help="also require the original Windows worktree bytes")
    args = parser.parse_args()
    if args.write:
        entries = []
        for p in paths():
            data = p.read_bytes()
            entries.append({"path": p.relative_to(ROOT).as_posix(),
                            "worktree_bytes": len(data),
                            "canonical_lf_bytes": len(canonical(data)),
                            "raw_worktree_sha256": digest(data),
                            "canonical_lf_sha256": digest(canonical(data)),
                            "preserve_raw_bytes": p.is_relative_to(OUT) and p.suffix in (".log", ".xml")})
        manifest = {"schema": "C1-03-evidence-v1", "date": "2026-09-29",
                    "policy": "SHA-256; canonical LF replaces CRLF with LF only. Raw worktree hashes describe the original delivery snapshot, not Git object IDs. Log/XML raw bytes are preserved by .gitattributes. Default verification accepts text newline normalization; --verify-working-bytes checks the original snapshot exactly.",
                    "self_exclusion": "Manifest excludes itself and SHA256SUMS; SHA256SUMS includes manifest but excludes itself.",
                    "files": entries}
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")
        lines = [e["canonical_lf_sha256"] + "  " + e["path"] for e in entries]
        lines.append(digest(canonical(MANIFEST.read_bytes())) + "  " + MANIFEST.relative_to(ROOT).as_posix())
        SUMS.write_text("\n".join(sorted(lines, key=lambda s: s.split("  ", 1)[1])) + "\n",
                        encoding="utf-8", newline="\n")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        data = (ROOT / entry["path"]).read_bytes()
        if digest(canonical(data)) != entry["canonical_lf_sha256"]:
            raise ValueError("canonical hash mismatch: " + entry["path"])
        if entry["preserve_raw_bytes"] or args.verify_working_bytes:
            if digest(data) != entry["raw_worktree_sha256"]:
                raise ValueError("raw hash mismatch: " + entry["path"])
    actual = {p.relative_to(ROOT).as_posix() for p in paths()}
    if actual != {e["path"] for e in manifest["files"]}:
        raise ValueError("manifest inventory mismatch")
    expected = {e["path"]: e["canonical_lf_sha256"] for e in manifest["files"]}
    expected[MANIFEST.relative_to(ROOT).as_posix()] = digest(canonical(MANIFEST.read_bytes()))
    listed = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        if name in listed:
            raise ValueError("duplicate checksum path: " + name)
        listed[name] = value
    if listed != expected:
        raise ValueError("SHA256SUMS mismatch")
    print(json.dumps({"status": "PASS", "files": len(actual),
                      "strict_worktree_bytes": args.verify_working_bytes,
                      "manifest_sha256": expected[MANIFEST.relative_to(ROOT).as_posix()]}))


if __name__ == "__main__":
    main()
