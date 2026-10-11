"""Verify a pre-Hybrid delivery without opening sealed data or training models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from historical_evidence import evidence_path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior evidence; select a new output')
    checks = []
    def check(name, path, expected):
        actual = sha(path) if path.is_file() else None
        checks.append(dict(name=name, path=str(path), expected=expected, actual=actual, passed=actual == expected))
    def read(name):
        return json.loads((ROOT / name).read_text(encoding='utf-8'))
    for manifest in ['experiments/C1-06R/training-freeze.json', 'experiments/C1-06S/final-freeze.json']:
        for name, expected in read(manifest)['files'].items():
            check(name, ROOT/name, expected)
    for manifest in ['experiments/C1-06S/evidence-manifest.json', 'experiments/PLAN-20261011-ROADMAP/artifact-manifest.json']:
        for item in read(manifest)['files']:
            check(item['path'], evidence_path(ROOT, item['path'], item['sha256']), item['sha256'])
    for manifest, base in [('experiments/C1-07/SHA256SUMS', 'experiments/C1-07/'), ('experiments/C1-07/DELIVERY_SHA256SUMS', '')]:
        for line in (ROOT/manifest).read_text(encoding='utf-8').splitlines():
            if line.strip():
                expected, name = line.split(maxsplit=1)
                name = base + name.strip().lstrip('*')
                check(name, evidence_path(ROOT, name, expected), expected)
    for item in read('experiments/PRE-HYBRID-20261011/historical-snapshots.json'):
        check(item['original'], ROOT/item['snapshot'], item['sha256'])
    for candidate in read('experiments/C1-07/HYBRID_HANDOFF.json')['candidates']:
        cp = candidate['checkpoint']; check(cp['path'], ROOT/cp['path'], cp['sha256'])
    doc = read('experiments/DOC-20261011-MODEL-PLAN/acceptance.json')
    check('unchanged original model use document', ROOT/'NeuroKinematics_Model_Kullanim_Plani_r1.docx', doc['source_sha256'])
    check('updated model use document', ROOT/'NeuroKinematics_Model_Kullanim_Plani_r2.docx', doc['output_sha256'])
    for name in ['validation', 'final1']:
        data = read('experiments/C1-06S/' + name + '.json')
        raw = 'data/generated/C1-06S/' + name + '/results.jsonl'
        check(raw, ROOT/raw, data['raw_sha256'])
    source = read('experiments/C1-06S/final1-inputs.json')
    check('geometric final requests', ROOT/'data/generated/C1-06S/final1-inputs/requests.json', source['requests_sha256'])
    # A changed expected checksum must never silently resolve to the old snapshot.
    check('resolver rejects different expected SHA', evidence_path(ROOT, 'README.md', '0'*64), sha(ROOT/'README.md'))
    handoff = read('experiments/C1-06S/HANDOFF.json')
    checks.append(dict(name='geometric engine is not learned', passed=handoff['learned'] is False))
    tasks = read('docs/roadmaps/ROADMAP_TASKS_r3.json')['tasks']
    checks.append(dict(name='future implementation not falsely accepted', passed=all(t['status']=='NOT_STARTED' for t in tasks)))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                  source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  checks=checks, count=len(checks), new_training='NOT_RUN', fresh_environment='NOT_CREATED',
                  sealed_access='OPAQUE_SHA_ONLY', independent_device_backup='NOT_CONFIRMED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if result['status'] != 'PASS':
        print(json.dumps([c for c in checks if not c['passed']]))
        sys.exit(1)


if __name__ == '__main__':
    main()
