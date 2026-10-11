"""Run selected regression directories separately, preserving every log/XML."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/C1-06S"
total = 0
for group, filename in (("c1_06s", "unit-r3.xml"), ("f0_01", "f001.xml"),
                        ("f0_02", "f002.xml"), ("f0_03", "f003.xml")):
    path = BASE / filename
    if path.exists():
        raise FileExistsError(path)
    subprocess.run([sys.executable, "-m", "pytest", "tests/"+group, "-q", "--junitxml="+str(path)],
                   cwd=ROOT, check=True)
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")):
        raise ValueError("test failure or skip")
    total += sum(int(s.attrib["tests"]) for s in suites)
print("PASS", total, "tests; isolated module namespaces", flush=True)
