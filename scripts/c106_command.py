"""Record one C1-06 command without overwriting previous attempts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace('-', '').replace('_', '').isalnum():
        parser.error('invalid command name')
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        parser.error('command required')
    output = ROOT / 'experiments/C1-06/commands' / args.name
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    env.update({key: '1' for key in keys})
    start = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except OSError as exc:
        code, stdout, stderr = 127, b'', str(exc).encode('utf-8')
    (output / 'stdout.log').write_bytes(stdout)
    (output / 'stderr.log').write_bytes(stderr)
    record = dict(argv=argv, cwd=str(ROOT), start_utc=start,
                  end_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
                  thread_environment={key: env[key] for key in keys},
                  stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                  stderr_sha256=hashlib.sha256(stderr).hexdigest())
    (output / 'command.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    sys.stdout.buffer.write(stdout)
    sys.stderr.buffer.write(stderr)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
