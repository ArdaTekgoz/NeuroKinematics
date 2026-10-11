"""Install verified restored artifacts; never overwrite existing project files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
from archive_core import sha, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists():
        raise FileExistsError('Receipt already exists')
    manifest = verify(a.archive.resolve())
    source_root = a.artifact_root.resolve()
    pending, existing = [], []
    for name, entry in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        source, target = (source_root/name).resolve(), (ROOT/name).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(ROOT):
            raise ValueError('Path escape')
        if not source.is_file() or sha(source) != entry['sha256']:
            raise ValueError('Restored artifact missing or changed: ' + name)
        if target.exists():
            if not target.is_file() or sha(target) != entry['sha256']:
                raise ValueError('Existing project artifact differs: ' + name)
            existing.append(name)
        else:
            pending.append((name, source, target, entry['sha256']))
    # All conflicts and source hashes are checked before copying any file.
    for name, source, target, expected in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as out:
            shutil.copyfileobj(src, out)
        if sha(target) != expected:
            raise ValueError('Installed hash differs: ' + name)
    result = dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),members=len(manifest['files']),
                  installed=len(pending),existing_identical=len(existing),overwrites=0,
                  source_root=str(source_root),project_root=str(ROOT),sealed_access='OPAQUE_BYTES_ONLY')
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    with a.receipt.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
