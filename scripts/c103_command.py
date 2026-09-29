"""Capture an actual C1-03 subprocess, with immutable per-attempt logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/C1-03/stage2')
    parser.add_argument('--id', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[0] == '--': command = command[1:]
    out = Path(args.output) / 'commands' / args.id
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    start = datetime.now(timezone.utc).isoformat()
    with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
    record = dict(argv=command, cwd=str(Path.cwd()), start_utc=start,
                  end_utc=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode,
                  threads=1, stdout='stdout.log', stderr='stderr.log')
    (out/'command.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(record))
    print((out/'stdout.log').read_text(encoding='utf-8', errors='replace')[-4000:])
    print((out/'stderr.log').read_text(encoding='utf-8', errors='replace')[-4000:])
    raise SystemExit(result.returncode)

if __name__ == '__main__': main()
