"""Read-only audit of Core evidence; new outputs never overwrite old runs."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from historical_evidence import evidence_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/PLAN-20261011-ROADMAP/audit'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(path, expected, group):
        p = evidence_path(ROOT, path, expected)
        actual = digest(p) if p.is_file() else None
        checks.append(dict(group=group, path=path, expected=expected, actual=actual, passed=actual == expected))
    for source in ['experiments/C1-06R/training-freeze.json', 'experiments/C1-06S/final-freeze.json']:
        for name, sha in read(source)['files'].items():
            check(name, sha, source)
    for entry in read('experiments/C1-06S/evidence-manifest.json')['files']:
        check(entry['path'], entry['sha256'], 'geometric_evidence')
    for source, base in [('experiments/C1-07/SHA256SUMS', 'experiments/C1-07/'),
                         ('experiments/C1-07/DELIVERY_SHA256SUMS', '')]:
        for line in (ROOT / source).read_text(encoding='utf-8').splitlines():
            if line.strip():
                sha, name = line.split(maxsplit=1)
                check(base + name.strip().lstrip('*'), sha, source)
    handoff = read('experiments/C1-07/HYBRID_HANDOFF.json')
    for candidate in handoff['candidates']:
        cp = candidate['checkpoint']
        check(cp['path'], cp['sha256'], 'neural_checkpoint')
    final = read('experiments/C1-06S/final1-audit.json')
    for name, key in [('data/generated/C1-06S/final1/results.jsonl', 'raw_sha256'),
                      ('data/generated/C1-06S/final1-inputs/requests.json', 'requests_sha256')]:
        check(name, final[key], 'final_raw')
    (OUT / 'integrity.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in checks),
        checks=checks), indent=2), encoding='utf-8')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    runs = []
    for group in ['c1_06s', 'f0_01', 'f0_02', 'f0_03']:
        command = [sys.executable, '-m', 'pytest', 'tests/' + group, '-q', '--junitxml=' + str(OUT / (group + '.xml'))]
        start = datetime.now(timezone.utc).isoformat()
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (OUT / (group + '.stdout.log')).write_text(run.stdout, encoding='utf-8')
        (OUT / (group + '.stderr.log')).write_text(run.stderr, encoding='utf-8')
        suites = list(ET.parse(OUT / (group + '.xml')).getroot().iter('testsuite'))
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'errors', 'failures', 'skipped']}
        runs.append(dict(group=group, command=command, started=start, finished=datetime.now(timezone.utc).isoformat(),
                         exit_code=run.returncode, **counts))
        print(group, counts, flush=True)
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version, platform=platform.platform(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        integrity_checks=len(checks), integrity_failures=[c for c in checks if not c['passed']],
        runs=runs, new_training='NOT_RUN', new_benchmark='NOT_RUN', fresh_environment='NOT_CREATED')
    result['passed'] = not result['integrity_failures'] and all(r['exit_code'] == 0 and not any(r[k] for k in ['errors','failures','skipped']) for r in runs)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('AUDIT', 'PASS' if result['passed'] else 'FAIL', 'integrity checks:', len(checks), flush=True)
    sys.exit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
