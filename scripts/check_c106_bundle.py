"""Verify archived C1-06 evidence without inference, tuning or new test access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from audit_c106_inputs import fingerprint
from check_c106_stage1 import verify_row

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/C1-06'
STAGE = BASE / 'stage2'
FINAL = STAGE / 'final-001'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def verify_sums(path):
    seen = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in seen, 'duplicate manifest path: ' + rel)
        seen.add(rel)
        verify_row(dict(path=rel, sha256=digest))
    return len(seen)


def verify_handoff(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            verify_row(value)
        for child in value.values():
            verify_handoff(child)
    elif isinstance(value, list):
        for child in value:
            verify_handoff(child)


def q_repeat():
    keys = ('query_id', 'group_id', 'subset', 'mode', 'pass_index', 'q_raw_rad',
            'shape_valid', 'finite', 'in_limits', 'profile_a', 'profile_b',
            'position_error_m', 'orientation_error_deg', 'failure_class')
    total = 0
    for seed in (2026100201, 2026100202, 2026100203):
        raw = ROOT / 'data/generated/C1-06/final-001'
        with (raw / f'E-C01-conditioned-seed-{seed}.jsonl').open() as old, \
                (raw / f'E-C03-Q-seed-{seed}.jsonl').open() as new:
            seen = set()
            for _ in range(12000):
                a, b = json.loads(next(old)), json.loads(next(new))
                require(a['pass_index'] == b['pass_index'] == 0, 'wrong Q pass')
                require(a['query_id'] not in seen, 'duplicate Q query')
                seen.add(a['query_id'])
                require(all(a[k] == b[k] for k in keys), 'Q replay differs')
                total += 1
    return total


def links():
    count = 0
    for path in (STAGE / 'RUN_REPORT.md', STAGE / 'RESULTS.md',
                 ROOT / 'docs/tasks/C1-06.md', ROOT / 'docs/roadmaps/C1_Core.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#', 1)[0].strip('<>')
            require((path.parent / target).exists(), f'broken link: {path}: {target}')
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = STAGE / 'evidence-manifest.json'
    sums = STAGE / 'SHA256SUMS'
    if args.freeze:
        require(not manifest.exists() and not sums.exists(), 'bundle already frozen')
    audit = read(BASE / 'input-hashes.json')
    require(audit['status'] == 'PASS', 'input audit failed')
    for row in audit['files']:
        verify_row(row)
    stage1_count = verify_sums(BASE / 'SHA256SUMS')
    preflight = read(STAGE / 'preflight.json')
    require(preflight['status'] == 'PASS', 'preflight failed')
    for row in preflight['source_files']:
        verify_row(row)
    suites = ET.parse(STAGE / 'runtime-tests.xml').getroot().iter('testsuite')
    totals = {k: 0 for k in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for k in totals:
            totals[k] += int(suite.get(k, 0))
    require(totals == dict(tests=59, failures=0, errors=0, skipped=0), 'test evidence differs')
    acceptance = read(FINAL / 'acceptance.json')
    require(acceptance['T_C05'] == 'PASS' and acceptance['H2'] == 'REJECTED', 'decision differs')
    for name, key in (('results.json', 'results_sha256'), ('raw-manifest.json', 'raw_manifest_sha256')):
        require(fingerprint(FINAL / name)['sha256'] == acceptance[key], 'acceptance digest differs')
    require(fingerprint(STAGE / 'preflight.json')['sha256'] == acceptance['stage2_preflight_sha256'],
            'preflight digest differs')
    raw = read(FINAL / 'raw-manifest.json')
    require(len(raw['files']) == 27, 'raw file count differs')
    for row in raw['files']:
        verify_row(row)
    require(sum(r['bytes'] for r in raw['files']) == raw['bytes'] == 2331485455, 'raw bytes differ')
    raw_rows = sum(r.get('rows', 0) for r in raw['files'])
    require(raw_rows == 1860000, 'measurement raw rows differ')
    diagnostics = read(ROOT / 'data/generated/C1-06/final-001/query-diagnostics.json')
    require(len(diagnostics) == len({r['query_id'] for r in diagnostics}) == 12000,
            'query diagnostics count differs')
    verify_handoff(read(FINAL / 'C1-07-handoff.json'))
    commands = {}
    for path in (BASE / 'commands').glob('*/command.json'):
        record = read(path)
        for stream in ('stdout', 'stderr'):
            require(fingerprint(path.parent / (stream + '.log'))['sha256'] == record[stream + '_sha256'],
                    'command log digest differs: ' + str(path))
        # Retain failed historical attempts, including the delivery checker schema repair.
        expected = 1 if path.parent.name in (
            'input-audit', 'stage2-c104-witness', 'stage2-bundle-freeze') else 0
        require(record['exit_code'] == expected, 'unexpected command failure: ' + str(path))
        commands[path.parent.name] = record['exit_code']
    for name in ('stage2-runtime-tests', 'stage2-preflight', 'stage2-identity',
                 'stage2-evaluate', 'stage2-summarize', 'stage2-audit'):
        require(commands.get(name) == 0, 'missing successful command: ' + name)
    report = dict(status='PASS', stage1_inputs=len(audit['files']), stage1_frozen_files=stage1_count,
                  tests=totals, raw_files=len(raw['files']), raw_bytes=raw['bytes'],
                  raw_rows=raw_rows, query_diagnostics=len(diagnostics), q_replay_pairs=q_repeat(),
                  markdown_links=links(), T_C05=acceptance['T_C05'], H2=acceptance['H2'],
                  new_inference='NOT_RUN', C1_02_test_decoding='NOT_RUN')
    if args.freeze:
        paths = {p for p in STAGE.rglob('*') if p.is_file()}
        paths |= {p for p in (BASE / 'commands').rglob('*') if p.is_file()}
        paths |= {ROOT / row['path'] for row in preflight['source_files']}
        paths.add(Path(__file__).resolve())
        rows = [dict(path=p.relative_to(ROOT).as_posix(), **fingerprint(p)) for p in sorted(paths)]
        payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), verification=report, files=rows,
                       note='Self and SHA256SUMS excluded; subsequent command logs checked dynamically. '
                            'Shared live task/status/roadmap documents are not immutable evidence.')
        with manifest.open('x', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(payload, indent=2) + '\n')
        rows.append(dict(path=manifest.relative_to(ROOT).as_posix(), **fingerprint(manifest)))
        with sums.open('x', encoding='utf-8', newline='\n') as f:
            for row in sorted(rows, key=lambda r: r['path']):
                f.write(f"{row['sha256']}  {row['path']}\n")
    for row in read(manifest)['files']:
        verify_row(row)
    report['bundle_frozen_files'] = verify_sums(sums)
    report['command_logs'] = len(commands)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
