"""Resolve explicit historical documents without relaxing SHA checks."""
import json
from pathlib import Path


def evidence_path(root: Path, name: str, expected: str) -> Path:
    mapping = root / 'experiments/PRE-HYBRID-20261011/historical-snapshots.json'
    if mapping.is_file():
        for entry in json.loads(mapping.read_text(encoding='utf-8')):
            if entry['original'] == name and entry['sha256'] == expected:
                path = (root / entry['snapshot']).resolve()
                if not path.is_relative_to(root.resolve() / 'docs/records/archive/pre-hybrid-20261011'):
                    raise ValueError('Historical snapshot outside approved archive')
                return path
    return root / name
