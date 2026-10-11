"""Publish a training-only freeze after checking all evidence, including failures."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/C1-06R'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = BASE/'training-freeze.json'
    if target.exists():
        raise FileExistsError('freeze already exists; do not silently overwrite it')
    checks = ['r0r1/attempt-001/preflight.json', 'r0r1/attempt-001/runtime.json',
              'r0r1/attempt-001/data-audit.json', 'r0r1/attempt-001/mixed64-result.json',
              'r0r1/diagnostic-r3/result.json', 'training-smoke.json']
    for name in checks:
        if read(BASE/name)['status'] != 'PASS':
            raise ValueError('failed evidence: '+name)
    if read(BASE/'r0r1/attempt-001/local64-result.json')['status'] != 'FAIL' or read(BASE/'r0r1/diagnostic-r2/result.json')['status'] != 'FAIL':
        raise ValueError('historical failed diagnostics must remain failures')
    suites = ['f0_01', 'f0_02', 'f0_03', 'c1_02', 'c1_03', 'c1_04', 'c1_05', 'c1_06r']
    xmls = [BASE/(x+'-regression.xml') for x in suites] + [BASE/'training-resume-tests.xml']
    total = 0
    for path in xmls:
        for suite in ET.parse(path).getroot().iter('testsuite'):
            if any(int(suite.get(key, '0')) for key in ('failures', 'errors', 'skipped')):
                raise ValueError('regression gate: '+str(path))
            total += int(suite.get('tests', '0'))
    smoke = read(BASE/'training-smoke.json')
    if smoke['code_sha256'] != digest(ROOT/'src/neurokinematics/neural/c106r_training.py') or smoke['protocol_sha256'] != digest(BASE/'training-round1.json'):
        raise ValueError('training code/protocol changed after smoke')
    paths = list((ROOT/'src/neurokinematics').rglob('*.py')) + list((ROOT/'tests/c1_06r').rglob('*.py'))
    paths += [ROOT/'scripts'/name for name in ['Start-C106RTraining.ps1','run_c106r_round1.py','check_c106r_training.py','c106r_command.py','freeze_c106r_training.py']]
    paths += [BASE/'training-round1.json', BASE/'r0r1-config.json', BASE/'requirements-win-cu128.lock', *xmls]
    paths += [BASE/name for name in checks]
    prior = read(BASE/'r0r1/attempt-001/preflight.json')
    for item in prior['files']:
        path = ROOT/item['path']
        if digest(path) != item['sha256']:
            raise ValueError('R0 input drift: '+str(path))
        paths.append(path)
    files = {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}
    record = dict(status='READY_FOR_USER_TRAINING', scope='VALIDATION_ROUND1_ONLY',
                  main_training='NOT_RUN', new_final_test='NOT_CREATED',
                  old_H2='REJECTED_UNCHANGED', product_target='NOT_MEASURED',
                  gpu=read(BASE/'r0r1/attempt-001/runtime.json')['gpu'], tests_passed=total,
                  diagnostic_failures_preserved=['attempt-001/local64 61/64', 'diagnostic-r2 local64 61/64'],
                  diagnostic_completion='mixed64 64/64; local64 diagnostic-r3 64/64 with separately registered optimizer scaling',
                  head_at_freeze=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), files=files)
    target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'], files=len(files), tests_passed=total)))


if __name__ == '__main__':
    main()
