"""Record official Foundations regression commands without rewriting past evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def run_commands(cwd, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    (output/'junit').mkdir(exist_ok=True)
    commands = [('lock', ['pixi', 'lock', '--check']),
                ('install', ['pixi', 'install', '--locked']),
                ('environment', ['pixi', 'run', '--locked', 'env-check', '--output', str(output/'environment.json')])]
    for i in range(5):
        commands.append((f'f0{i}', ['pixi', 'run', '--locked', f'test-f0{i}']))
    commands += [
        ('f05-unit', ['pixi', 'run', '--locked', 'python', '-m', 'pytest', '-q', 'tests/f0_05', '--ignore=tests/f0_05/test_tf08.py', '--ignore=tests/f0_05/test_mutations.py', '--ignore=tests/f0_05/test_stage2_mutations.py']),
        ('tf08', ['pixi', 'run', '--locked', 'python', '-m', 'pytest', '-q', 'tests/f0_05/test_tf08.py']),
        ('f05-mutations', ['pixi', 'run', '--locked', 'python', '-m', 'pytest', '-q', 'tests/f0_05/test_mutations.py', 'tests/f0_05/test_stage2_mutations.py'])]
    if (Path(cwd)/'tests/f0_06').is_dir():
        commands.append(('f06', ['pixi', 'run', '--locked', 'python', '-m', 'pytest', '-q', 'tests/f0_06']))
    records = []
    for name, argv in commands:
        junit = output/'junit'/f'{name}.xml'
        if name.startswith('f0') or name == 'tf08':
            argv += [f'--junitxml={junit}', '-p', 'no:cacheprovider', '-o', 'junit_family=legacy']
        entry = {'name': name, 'argv': argv, 'command': subprocess.list2cmdline(argv), 'cwd': str(Path(cwd).resolve()), 'started_utc': now()}
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True)
        entry.update(finished_utc=now(), exit_code=result.returncode)
        for kind, content in [('stdout', result.stdout), ('stderr', result.stderr)]:
            path = output/'logs'/f'{name}.{kind}.log'
            path.write_bytes(content)
            entry[kind] = {'path': str(path), 'sha256': sha(path)}
        if junit.exists():
            suites = ET.parse(junit).getroot().findall('testsuite')
            counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
            counts['passed'] = counts['tests']-counts['failures']-counts['errors']-counts['skipped']
            entry['junit'] = {'path': str(junit), 'sha256': sha(junit), **counts}
        records.append(entry)
        write(output/'commands.json', records)
        print(name, result.returncode, entry.get('junit', {}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cwd', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run_commands(args.cwd, args.output)
