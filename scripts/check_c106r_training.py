"""Read-only handoff gate; no model training or final evaluation."""
import importlib.metadata as metadata
from pathlib import Path
import shutil
import torch
from neurokinematics.neural.c104 import ROOT, read_json
from neurokinematics.neural.c106r import configure, guard_hashes


def check():
    configure()
    freeze = read_json(ROOT / "experiments/C1-06R/training-freeze.json")
    if freeze["status"] != "READY_FOR_USER_TRAINING":
        raise ValueError("handoff is not ready")
    guard_hashes(freeze["files"])
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != freeze["gpu"]:
        raise ValueError("different or unavailable GPU; revalidate the runtime")
    runtime = read_json(ROOT / "experiments/C1-06R/r0r1/attempt-001/runtime.json")
    for name, expected in runtime["packages"].items():
        dist = metadata.distribution(name)
        import json
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if dist.version != expected["version"] or direct != expected["direct_url"]:
            raise ValueError("installed artifact drift: " + name)
    if shutil.disk_usage(ROOT).free < 5*1024**3:
        raise ValueError("at least 5 GiB free space required for campaign records")
    print("PASS: frozen inputs, exact runtime artifacts, GPU and disk. Long training NOT_RUN by this check.", flush=True)
    return freeze


if __name__ == "__main__":
    check()
