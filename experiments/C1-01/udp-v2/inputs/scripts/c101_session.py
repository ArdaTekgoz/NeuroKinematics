"""One recorded runtime for C1-01 prepare -> smoke -> pilot -> full -> verify.

Run through run_c101_session.ps1. Inputs are snapshotted before preparation;
this module never rewrites frozen Foundations files or earlier run evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.benchmark.queries import encode_query
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core.contract import ROOT, load_contract, sha256
from neurokinematics.core.diagnostics import probe_expired_requests
from neurokinematics.core.results import validate_result_record
from neurokinematics.core.runner import (FROZEN_QUERY_PATH, SOLVER_IDS, load_queries,
    run_method, smoke_selection, summarize_file, utc_now, write_json)

WORKER = Path('/opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker')
ENVIRONMENT = {'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp', 'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'NUMEXPR_NUM_THREADS': '1', 'ROS_DISTRO': 'jazzy', 'RCUTILS_LOGGING_USE_STDOUT': '0',
               'PIXI_NO_INSTALL': 'true'}
FATAL = {'PROCESS_FAILURE', 'INSTALLATION_FAILURE', 'ADAPTER_ERROR', 'INVALID_OUTPUT', 'VALIDATION_ERROR'}
DESELECT = 'tests/f0_06/test_gate.py::test_real_benchmark_schema_mutation'
MEMORY_LIMIT_BYTES = 8 * 1024 ** 3


def memory_observation(path=Path('/proc/meminfo')):
    values = {k: v.strip() for k, v in (line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)}
    return {'mem_total': values['MemTotal'], 'mem_available': values.get('MemAvailable', 'NOT_AVAILABLE')}


def memory_policy(root=Path('/sys/fs/cgroup')):
    # Docker cgroup v2 enforces the container ceiling; /proc/meminfo describes the host.
    try:
        limit = (root / 'memory.max').read_text().strip()
        swap = (root / 'memory.swap.max').read_text().strip()
    except FileNotFoundError as exc:
        raise ValueError('cgroup v2 memory/swap limits required') from exc
    if limit != str(MEMORY_LIMIT_BYTES) or swap != '0':
        raise ValueError('8 GiB memory ceiling with zero swap required; got ' + limit + '/' + swap)
    return {'controller': 'cgroup-v2', 'memory_max_bytes': MEMORY_LIMIT_BYTES, 'swap_max_bytes': 0}


def read_json(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'))


def command_output(args):
    return subprocess.check_output(args, text=True, encoding='utf-8', stderr=subprocess.STDOUT).strip()


def tree_hashes(path, suffix='*.py'):
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob(suffix)) if p.is_file()}


def require_policy(system, machine, release, affinity, environment):
    if (system != 'Linux' or machine != 'x86_64' or release.get('ID') != 'ubuntu'
            or release.get('VERSION_ID') != '24.04' or affinity != [0, 1]):
        raise ValueError('Ubuntu 24.04 x86_64 / CPU [0,1] required')
    if environment != ENVIRONMENT:
        raise ValueError('runtime environment differs from the locked UDPv4/thread policy')


def capture_runtime(session, image_id):
    config = load_contract()
    base_path = session / 'base-environment-lock.json'
    base = read_json(base_path)
    release = {k: v.strip('"') for k, v in (line.split('=', 1) for line in
               Path('/etc/os-release').read_text().splitlines() if '=' in line)}
    affinity = sorted(os.sched_getaffinity(0))
    environment = {name: os.environ.get(name) for name in ENVIRONMENT}
    require_policy(platform.system(), platform.machine(), release, affinity, environment)
    if image_id != base['image_id'] or not image_id.startswith('sha256:'):
        raise ValueError('image identity differs from build audit')
    packages = command_output(['dpkg-query', '-W', '-f=${binary:Package}\t${Version}\t${Architecture}\n'])
    closure = hashlib.sha256(packages.encode()).hexdigest()
    if closure != base['dpkg_closure_sha256'] or sha256(ROOT / 'pixi.lock') != base['pixi_lock_sha256']:
        raise ValueError('installed package closure or Pixi lock changed')
    sources = {}
    for name, expected in base['sources'].items():
        path = '/opt/c101/external/' + name
        commit = command_output(['git', '-C', path, 'rev-parse', 'HEAD'])
        if commit != expected['commit'] or command_output(['git', '-C', path, 'status', '--porcelain']):
            raise ValueError('external source changed: ' + name)
        sources[name] = commit
    cpu = [{k.strip(): v.strip() for k, v in (line.split(':', 1) for line in block.splitlines()
            if ':' in line) if k.strip() in ('processor', 'vendor_id', 'model name', 'cpu family', 'model', 'flags')}
           for block in Path('/proc/cpuinfo').read_text().strip().split('\n\n')]
    memory = memory_policy()
    if int(memory_observation()['mem_total'].split()[0]) * 1024 < MEMORY_LIMIT_BYTES:
        raise ValueError('host RAM is below the locked 8 GiB container ceiling')
    governors = {p.parent.parent.name: p.read_text().strip() for p in
                 Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor')}
    scripts = Path(__file__).resolve().parent
    return {'schema_version': '1.1.0', 'image_id': image_id, 'base_environment_lock_sha256': sha256(base_path),
            'dpkg_closure_sha256': closure, 'pixi_lock_sha256': sha256(ROOT / 'pixi.lock'),
            'sources': sources, 'environment': environment, 'cpu_affinity': affinity,
            'os_release': release, 'kernel': platform.release(), 'cpu': cpu, 'memory_policy': memory,
            'cpu_governors': governors or 'NOT_AVAILABLE',
            'python': platform.python_version(), 'compiler': base['compiler'], 'libc': base['libc'],
            'source_files': tree_hashes(ROOT / 'src'), 'test_files': tree_hashes(ROOT / 'tests'),
            'scripts': tree_hashes(scripts, '*'), 'worker_binary_sha256': sha256(WORKER),
            'compiled_worker_source': tree_hashes(ROOT / 'ros2_ws/src', '*'),
            'schema_sha256': sha256(ROOT / 'experiments/C1-01/result-schema.json'),
            'pytest_config_sha256': sha256(ROOT / 'pyproject.toml'),
            'baseline_config_sha256': sha256(ROOT / 'experiments/C1-01/baseline-config.json'),
            'query_list_sha256': config['queries']['list_sha256']}


def pilot_selection(rows):
    counts, selected = Counter(), []
    expected = {(subset, start) for subset in ('main', 'boundary', 'singularity') for start in ('local', 'wide')}
    for row in rows:
        key = row['subset'], row['start_class']
        if key in expected and counts[key] < 2:
            selected.append(row)
            counts[key] += 1
    if set(counts) != expected or any(counts[k] != 2 for k in expected):
        raise ValueError('pilot requires two frozen queries from each of six groups')
    return selected


def xml_result(path, nodeids):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('.//testsuite')
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ('errors', 'failures', 'skipped')):
        raise ValueError('regression has failures/errors/skips')
    actual = [(case.attrib['classname'], case.attrib['name']) for s in suites for case in s.findall('testcase')]
    expected = []
    for node in nodeids:
        parts = node.split('::')
        expected.append(('.'.join([parts[0][:-3].replace('/', '.'), *parts[1:-1]]), parts[-1]))
    if not expected or Counter(actual) != Counter(expected):
        raise ValueError('JUnit test identities differ from collected tests')
    return len(actual)


def check_gate(session, stage, runtime_sha, *, allowed=('PASS',)):
    path = session / stage / 'gate.json'
    gate = read_json(path)
    if gate.get('status') not in allowed or gate.get('runtime_sha256') != runtime_sha or gate.get('stage') != stage:
        raise ValueError('gate status/runtime mismatch: ' + stage)
    for name, expected in gate['files'].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent.resolve()) or not target.is_file() or sha256(target) != expected:
            raise ValueError('gate evidence missing or changed: ' + stage + '/' + name)
    if stage in ('smoke', 'pilot', 'full'):
        check_gate(session, 'prepare', runtime_sha)
        if gate.get('prepare_gate_sha256') != sha256(session / 'prepare/gate.json'):
            raise ValueError('prepare gate binding changed: ' + stage)
    previous = {'pilot': 'smoke', 'full': 'pilot', 'verify': 'full'}.get(stage)
    if previous:
        check_gate(session, previous, runtime_sha,
                   allowed=('MEASURED_UNVERIFIED',) if previous == 'full' else ('PASS',))
        field = 'full_gate_sha256' if stage == 'verify' else 'previous_gate_sha256'
        if gate.get(field) != sha256(session / previous / 'gate.json'):
            raise ValueError('previous gate binding changed: ' + stage)
    return gate


def verify_small(raw, solver, config, queries, manifest, stage):
    validator = CandidateValidator()
    counts = Counter()
    easy = 0
    with raw.open('rb') as stream:
        for deadline in ((50,) if stage == 'smoke' else (10, 50)):
            for query in queries:
                payload = stream.readline()
                if not payload:
                    raise ValueError('missing measured row')
                record = strict_json(payload.decode('utf-8'))
                if (encode_query(record) != payload or record['deadline_profile_ms'] != deadline
                        or record['measurement_pass_index'] != 0):
                    raise ValueError('measured row order/encoding mismatch')
                validate_result_record(record, query, solver, config, config['queries']['list_sha256'],
                                       manifest['dataset_manifest_sha256'], validator)
                counts[record['common_status']] += 1
                easy += bool(record['subset'] == 'main' and record['profile_b_deadline'])
        if stream.readline():
            raise ValueError('extra measured rows')
    return counts, easy


def preparation(session, runtime_sha):
    output = session / 'prepare'
    output.mkdir()
    arguments = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 'tests/c1_01', 'tests/f0_05', 'tests/f0_06', '--deselect=' + DESELECT]
    collect = subprocess.run([*arguments, '--collect-only'], cwd=ROOT, text=True, capture_output=True)
    (output / 'collection.log').write_text(collect.stdout + collect.stderr, encoding='utf-8')
    if collect.returncode:
        raise ValueError('pytest collection failed; see collection.log')
    nodeids = [line.strip() for line in collect.stdout.splitlines() if line.startswith('tests/') and '::' in line]
    write_json(output / 'collected-tests.json', {'nodeids': nodeids, 'deselected': DESELECT, 'decision': 'ADR-008'})
    with (output / 'pytest.log').open('w', encoding='utf-8') as log:
        tested = subprocess.run([*arguments, '-o', 'junit_family=legacy',
                                '--junitxml=' + str(output / 'regression.xml')], cwd=ROOT,
                                text=True, stdout=log, stderr=subprocess.STDOUT)
    print((output / 'pytest.log').read_text(encoding='utf-8'), flush=True)
    if tested.returncode:
        raise ValueError('critical regression failed')
    count = xml_result(output / 'regression.xml', nodeids)
    config = load_contract()
    rows, _ = load_queries(FROZEN_QUERY_PATH, config)
    probe = probe_expired_requests(config, rows, output / 'protocol', WORKER)
    if probe['status'] != 'PASS':
        raise ValueError('actual worker expiry protocol failed; see protocol/expired-request-probe.json')
    files = {name: sha256(output / name) for name in
             ('regression.xml', 'collected-tests.json', 'pytest.log', 'collection.log')}
    files.update({'protocol/' + name: digest for name, digest in tree_hashes(output / 'protocol', '*').items()})
    return {'stage': 'prepare', 'status': 'PASS', 'runtime_sha256': runtime_sha,
            'test_count': count, 'deselected': DESELECT, 'expired_request_probe': probe['status'],
            'files': files}


def measurement(session, stage, runtime_sha):
    check_gate(session, 'prepare', runtime_sha)
    tests = read_json(session / 'prepare/collected-tests.json')['nodeids']
    xml_result(session / 'prepare/regression.xml', tests)
    config = load_contract()
    rows, manifest = load_queries(FROZEN_QUERY_PATH, config)
    previous = None
    prerequisites = () if stage == 'smoke' else ('smoke',) if stage == 'pilot' else ('smoke', 'pilot')
    for prerequisite in prerequisites:
        previous = prerequisite
        gate = check_gate(session, previous, runtime_sha)
        if set(gate['solvers']) != set(SOLVER_IDS):
            raise ValueError('gate missing a mandatory solver')
        small = smoke_selection(rows) if previous == 'smoke' else pilot_selection(rows)
        for solver in config['solvers']:
            raw = session / previous / (solver['id'].replace('/', '-') + ('-smoke.jsonl' if previous == 'smoke' else '-benchmark.jsonl'))
            counts, easy = verify_small(raw, solver, config, small, manifest, previous)
            if any(counts[k] for k in FATAL) or (previous == 'smoke' and not easy):
                raise ValueError('previous gate does not pass independent review')
    output = session / stage
    output.mkdir()
    chosen = smoke_selection(rows) if stage == 'smoke' else pilot_selection(rows) if stage == 'pilot' else rows
    plan = copy.deepcopy(config)
    if stage != 'full':
        plan['benchmark']['measurement_passes'] = 1
    mode = 'smoke' if stage == 'smoke' else 'benchmark'
    expected = len(chosen) * (1 if stage == 'smoke' else 2) * plan['benchmark']['measurement_passes']
    result = {'stage': stage, 'status': 'IN_PROGRESS', 'runtime_sha256': runtime_sha,
              'started_utc': utc_now(), 'solvers': {}, 'files': {},
              'prepare_gate_sha256': sha256(session / 'prepare/gate.json'),
              'previous_gate_sha256': sha256(session / previous / 'gate.json') if previous else None,
              'query_list_sha256': config['queries']['list_sha256'], 'expected_per_solver': expected}
    write_json(output / 'gate.json', result)
    for solver in config['solvers']:
        print(f"START {stage}: {solver['id']} ({expected} attempts)", flush=True)
        def progress(info):
            write_json(output / 'progress.json', {**info, 'updated_utc': utc_now()})
            print(json.dumps(info), flush=True)
        summary = run_method(solver, plan, chosen, manifest, output, WORKER, rows[:20], mode=mode, progress_callback=progress)
        raw = Path(summary['raw_path'])
        complete = summary['record_count'] == expected and summary['worker_start_error'] is None
        warm = summary['warmup_calls'] == 20 * len(summary['worker_launch_elapsed_ns'])
        passed = complete and warm
        if complete and stage != 'full':
            counts, easy = verify_small(raw, solver, config, chosen, manifest, stage)
            passed = passed and not any(counts[k] for k in FATAL) and (stage != 'smoke' or easy > 0)
        state = ('MEASURED_UNVERIFIED' if stage == 'full' else 'PASS') if passed else 'INCOMPLETE'
        result['solvers'][solver['id']] = {'status': state, 'record_count': summary['record_count'],
                                          'raw_sha256': sha256(raw), 'raw_bytes': raw.stat().st_size,
                                          'raw_file': raw.name, 'storage_state': 'LOCAL_ONLY'}
        for path in (raw, output / (solver['id'].replace('/', '-') + '-' + mode + '-summary.json'),
                     Path(summary['stderr_path'])):
            if path.is_file():
                result['files'][path.name] = sha256(path)
        write_json(output / 'gate.json', result)
        if not passed:
            break
    result['finished_utc'] = utc_now()
    result['status'] = ('MEASURED_UNVERIFIED' if stage == 'full' else 'PASS') if (
        len(result['solvers']) == 5 and all(v['status'] in ('PASS', 'MEASURED_UNVERIFIED')
                                         for v in result['solvers'].values())) else 'INCOMPLETE'
    return result


def verification(session, runtime_sha):
    gate = check_gate(session, 'full', runtime_sha, allowed=('MEASURED_UNVERIFIED',))
    if set(gate['solvers']) != set(SOLVER_IDS):
        raise ValueError('incomplete full benchmark')
    output = session / 'verify'
    output.mkdir()
    result = {'stage': 'verify', 'status': 'PASS', 'runtime_sha256': runtime_sha,
              'full_gate_sha256': sha256(session / 'full/gate.json'), 'files': {}, 'solvers': {},
              'scope': 'record integrity; task acceptance requires documentation and requirement review'}
    for solver_id in SOLVER_IDS:
        safe = solver_id.replace('/', '-')
        summary = summarize_file(session / 'full' / (safe + '-benchmark.jsonl'), solver_id, 'benchmark', FROZEN_QUERY_PATH)
        path = output / (safe + '-summary.json')
        write_json(path, summary)
        result['files'][path.name] = sha256(path)
        counts = summary['groups']['all']['common_status_counts']
        fatal_count = sum(counts.get(kind, 0) for kind in FATAL)
        state = 'FAIL' if fatal_count else summary['status']
        result['solvers'][solver_id] = {'status': state, 'record_count': summary['record_count'],
                                      'fatal_infrastructure_count': fatal_count}
        if state != 'PASS':
            result['status'] = 'FAIL'
        print('VERIFIED ' + solver_id, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'pilot', 'full', 'verify'))
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    session = args.session
    mutex = session.parent / '.c101-active'
    with mutex.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps({'stage': args.stage, 'started_utc': utc_now()}))
    try:
        if (session / args.stage).exists():
            raise ValueError('stage evidence exists; use a new session instead of overwriting')
        memory_start = memory_observation()
        runtime = capture_runtime(session, args.image_id)
        lock = session / 'runtime-lock.json'
        if args.stage == 'prepare':
            if lock.exists():
                raise ValueError('runtime lock already exists')
            write_json(lock, runtime)
        elif read_json(lock) != runtime:
            expected = read_json(lock)
            changed = [key for key in sorted(expected.keys() | runtime.keys()) if expected.get(key) != runtime.get(key)]
            raise ValueError('runtime/code/test/environment drift since preparation: ' + ', '.join(changed))
        runtime_sha = sha256(lock)
        result = (preparation(session, runtime_sha) if args.stage == 'prepare' else
                  verification(session, runtime_sha) if args.stage == 'verify' else
                  measurement(session, args.stage, runtime_sha))
        if capture_runtime(session, args.image_id) != runtime:
            raise ValueError('runtime changed during stage; evidence cannot pass')
        result['host_memory_observations'] = {'start': memory_start, 'end': memory_observation()}
        write_json(session / args.stage / 'gate.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result['status'] in ('PASS', 'MEASURED_UNVERIFIED') else 1
    except Exception as exc:
        write_json(session / (args.stage + '-failure-' + uuid.uuid4().hex + '.json'),
                   {'status': 'FAIL', 'error': str(exc), 'at': utc_now()})
        raise
    finally:
        mutex.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
