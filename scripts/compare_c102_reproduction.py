"""Compare two independent C1-02 productions without treating wall times as data."""

import hashlib
import json
from pathlib import Path

from neurokinematics.data.pairs import EVIDENCE, ROOT, sha


def candidate_semantic_hash(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            item.pop("elapsed_ns")
            digest.update((json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
            count += 1
    return digest.hexdigest(), count


def main():
    directories = [ROOT / "data/generated/C1-02/v1", ROOT / "data/generated/C1-02/v1-repro"]
    manifests = [json.loads((directory / "dataset-manifest.json").read_text(encoding="utf-8")) for directory in directories]
    shard_equal = []
    for left, right in zip(manifests[0]["shards"], manifests[1]["shards"], strict=True):
        if left["path"] != right["path"] or left["record_count"] != right["record_count"]:
            raise ValueError("shard topology differs")
        shard_equal.append(left["file_sha256"] == right["file_sha256"] and
                           left["content_sha256"] == right["content_sha256"])
    candidate = [candidate_semantic_hash(directory / "teacher-candidates.jsonl") for directory in directories]
    report = {
        "status": "PASS" if all(shard_equal) and manifests[0]["dataset_content_sha256"] == manifests[1]["dataset_content_sha256"] and candidate[0] == candidate[1] else "FAIL",
        "run_paths": [directory.relative_to(ROOT).as_posix() for directory in directories],
        "dataset_content_sha256": [manifest["dataset_content_sha256"] for manifest in manifests],
        "shards": len(shard_equal), "equal_file_and_content_shards": sum(shard_equal),
        "candidate_semantic_sha256_excluding_elapsed_ns": [item[0] for item in candidate],
        "candidate_rows": [item[1] for item in candidate],
        "candidate_raw_file_sha256": [sha(directory / "teacher-candidates.jsonl") for directory in directories],
        "timing_excluded_reason": "Elapsed nanoseconds are measured wall time and not part of canonical pair content."
    }
    (EVIDENCE / "determinism-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "equal_shards": report["equal_file_and_content_shards"], "shards": report["shards"]}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
