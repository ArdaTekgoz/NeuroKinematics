"""Record each C1-07 subprocess attempt without replacing previous evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():parser.error('safe name required')
    argv=args.command[1:] if args.command[:1]==['--'] else args.command
    if not argv:parser.error('command required')
    root=Path(__file__).resolve().parents[1];out=root/'experiments/C1-07/commands'/args.name
    out.mkdir(parents=True,exist_ok=False);env=os.environ.copy()
    chosen={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    chosen.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUTF8='1');env.update(chosen)
    start=datetime.now(timezone.utc).isoformat()
    with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        try:code=subprocess.run(argv,cwd=root,env=env,stdout=stdout,stderr=stderr).returncode
        except OSError as exc:code=127;stderr.write(str(exc).encode('utf-8'))
    record=dict(argv=argv,cwd=str(root),start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),exit_code=code,environment=chosen)
    for key in ('stdout','stderr'):record[key+'_sha256']=hashlib.sha256((out/(key+'.log')).read_bytes()).hexdigest()
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record))
    if code:print((out/'stderr.log').read_text(encoding='utf-8',errors='replace')[-5000:])
    raise SystemExit(code)


if __name__=='__main__':main()
