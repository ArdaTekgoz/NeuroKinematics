"""Record the native runtime used for C1-02 generation and validation."""

import json
import os
from pathlib import Path
import platform
import sys

import numpy
import pinocchio
import scipy

from neurokinematics.data.pairs import EVIDENCE, ROOT, sha


def main():
    result = {
        "platform": platform.platform(), "system": platform.system(),
        "machine": platform.machine(), "processor": platform.processor(),
        "python": sys.version, "numpy": numpy.__version__,
        "scipy": scipy.__version__, "pinocchio": pinocchio.__version__,
        "pixi_lock_sha256": sha(ROOT / "pixi.lock"),
        "generation_source_sha256": sha(ROOT / "src/neurokinematics/data/pairs.py"),
        "validation_source_sha256": sha(ROOT / "src/neurokinematics/data/pair_validation.py"),
        "pilot_script_sha256": sha(ROOT / "scripts/run_c102_pilot.py"),
        "full_script_sha256": sha(ROOT / "scripts/run_c102_full.py"),
        "thread_environment": {name: os.getenv(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "gpu": "NOT_USED", "ros_docker": "NOT_USED",
    }
    (EVIDENCE / "environment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "RECORDED", "python": platform.python_version(), "pinocchio": pinocchio.__version__}))


if __name__ == "__main__":
    main()
