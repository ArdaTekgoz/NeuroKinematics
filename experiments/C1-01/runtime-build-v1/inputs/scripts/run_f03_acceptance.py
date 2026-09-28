"""Frozen F0-03 closure. Stop on failure; preserve actual output and exit codes.

--output temp/f03-prepush reruns the gate without changing committed evidence.
--verify-only checks an existing SHA256SUMS without regenerating it.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def verify_hashes(output):
    lines = (output/'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError('empty SHA256SUMS')
    for line in lines:
        expected, relative = line.split('  ', 1)
        actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'evidence hash mismatch: {relative}')
    return len(lines)


def hashes(output):
    # Narrative changes in the second commit must not invalidate numerical evidence.
    # Git protects the Markdown narrative; SHA256SUMS covers all other evidence.
    files = [p for p in output.iterdir() if p.is_file() and p.suffix != '.md' and p.name != 'SHA256SUMS']
    for folder in ('src/neurokinematics/kinematics', 'tests/f0_03', 'tests/f0_02'):
        files.extend((ROOT/folder).glob('*.py'))
    files.extend(ROOT/p for p in ('pixi.toml', 'pixi.lock', '.gitattributes', 'scripts/run_f03_acceptance.py'))
    lines = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}' for p in sorted(set(files))]
    (output/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def preflight():
    from neurokinematics.kinematics.model import load_robot
    from neurokinematics.kinematics.validation import sample_configurations, sample_hash
    inputs = load_robot()
    sample = sample_hash(sample_configurations(inputs))
    if sample != '8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101':
        raise ValueError('F0-02 sample hash mismatch')
    ancestors = ['d92dd213bb96f8932bd0019541dd13dd7365afaf', '34947308d31ccaf55bd8f74640bc12d6834dc605']
    for commit in ancestors:
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], cwd=ROOT, check=True)
    return {'status': 'PASS', 'input_hashes': inputs.hashes, 'f02_sample_sha256': sample,
            'remote_ancestors': ancestors, 'head_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'authorized_tcp_hash': inputs.hashes['config/robots/tcp_tool0.json'],
            'correction': 'User authorized the 64-character F0-02 hash; prior 62-character request was a typo, not an input change.'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/F0-03'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    output.relative_to(ROOT)  # evidence paths must remain within this checkout
    if args.verify_only:
        print(f'SHA256SUMS: {verify_hashes(output)} files verified')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'authorized-preflight.json', preflight())
    def junit(name):
        return f'--junitxml={(output/name).relative_to(ROOT).as_posix()}'
    target = str(output.relative_to(ROOT))
    commands = [
        ['pixi', 'lock', '--check'], ['pixi', 'install', '--locked'],
        ['pixi', 'run', '--locked', 'test-f00', junit('f00-junit.xml')],
        ['pixi', 'run', '--locked', 'verify-robot-a', '--output', str(output/'robot-verification.json')],
        ['pixi', 'run', '--locked', 'test-f01', junit('f01-junit.xml')],
        ['pixi', 'run', '--locked', 'test-f02', junit('f02-junit.xml'), '-o', 'junit_family=legacy'],
        ['pixi', 'run', '--locked', 'test-f03-unit', junit('unit-junit.xml'), '-o', 'junit_family=legacy'],
        ['pixi', 'run', '--locked', 'validate-jacobian', '--output', target],
        ['pixi', 'run', '--locked', 'validate-metrics', '--output', target],
        ['pixi', 'run', '--locked', 'test-f03', junit('pytest-junit.xml'), '-o', 'junit_family=legacy'],
    ]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
    records = []
    for command in commands:
        start = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        records.append({'command': command, 'exit_code': result.returncode, 'started_utc': start,
                        'finished_utc': datetime.now(timezone.utc).isoformat(), 'stdout': result.stdout, 'stderr': result.stderr})
        write_json(output/'commands.json', records)
        print('$ '+' '.join(command), flush=True)
        print(result.stdout+result.stderr, flush=True)
        if result.returncode:
            write_json(output/'acceptance.json', {'status': 'FAIL', 'failed_command': command})
            return result.returncode
    counts = {}
    for name in ('f00', 'f01', 'f02', 'unit', 'pytest', 'metrics'):
        cases = list(ET.parse(output/f'{name}-junit.xml').getroot().iter('testcase'))
        bad = [c.get('name') for c in cases if any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped'))]
        if not cases or bad:
            raise ValueError(f'incomplete JUnit evidence: {name}: {bad}')
        counts[name] = len(cases)
    cases = ET.parse(output/'pytest-junit.xml').getroot().iter('testcase')
    mutations = []
    for case in cases:
        properties = {p.get('name'): p.get('value') for p in case.findall('properties/property')}
        if 'mutation' in properties:
            mutations.append(properties)
    if len(mutations) != 12 or any(p['detected'] != 'True' for p in mutations):
        raise ValueError('all twelve mutations must be detected')
    write_json(output/'mutation-results.json', {'status': 'PASS', 'detected_count': len(mutations), 'cases': mutations})
    summary = json.loads((output/'jacobian-validation-summary.json').read_text(encoding='utf-8'))
    metrics = json.loads((output/'metric-validation-summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'PASS' or metrics['status'] != 'PASS':
        raise ValueError('T-F03 and T-F04 must both pass')
    write_json(output/'acceptance.json', {'status': 'PASS', 'tests': counts, 'command_count': len(records),
               'mutation_count': len(mutations), 'sample_sha256': summary['sample_sha256'],
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'preflight': preflight()})
    hashes(output)
    print(f'F0-03 PASS; SHA256SUMS: {verify_hashes(output)} files verified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
