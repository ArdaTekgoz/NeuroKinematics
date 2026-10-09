"""Resolve the official exact CUDA wheel hash, preserving the G0/CPU locks."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import unquote, urlparse, parse_qs, urljoin
import json

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://download.pytorch.org/whl/cu128/torch/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href")


def main():
    parser = Links()
    parser.feed(urlopen(INDEX, timeout=60).read().decode())
    found = [urljoin(INDEX, x) for x in parser.links if unquote(urlparse(x).path).endswith("/torch-2.10.0+cu128-cp312-cp312-win_amd64.whl")]
    if len(found) != 1:
        raise ValueError(f"expected exactly one official wheel, found {len(found)}")
    parsed = urlparse(found[0])
    sha = parse_qs(parsed.fragment)["sha256"][0]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid published SHA256")
    output = ROOT / "experiments/C1-06R"
    lock = output / "requirements-win-cu128.lock"
    if lock.exists():
        raise FileExistsError(lock)
    lines = ["# C1-06R Windows CPython 3.12 CUDA overlay; ADR-014.", "--only-binary=:all:"]
    lines.append(f"torch @ {parsed._replace(fragment='').geturl()} --hash=sha256:{sha}")
    for source in ("experiments/C1-03/requirements-win-cpu.lock", "experiments/C1-03/stage2/runtime-supplement.lock"):
        lines.extend(line for line in (ROOT / source).read_text().splitlines()
                     if " @ " in line and not line.startswith("torch @ "))
    lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = dict(index=INDEX, wheel=found[0], sha256=sha,
                  rationale="Same Torch 2.10.0 as historical CPU overlay; official CUDA 12.8 Blackwell build",
                  sources=["https://pytorch.org/get-started/previous-versions/", "https://pytorch.org/blog/pytorch-2-7/"],
                  runtime_verification="NOT_RUN")
    (output / "runtime-resolution.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
