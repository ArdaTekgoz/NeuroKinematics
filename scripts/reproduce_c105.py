"""Fresh locked environment and C1-05 inference verification in a clean checkout."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--witness',type=Path,required=True);a=p.parse_args()
    checkout=a.checkout.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if (checkout/'.pixi').exists() or (checkout/'.venv').exists():raise ValueError('fresh environment required')
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True)
    if dirty.strip():raise ValueError('clean checkout required')
    def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    write(out/'start.json',dict(head=head,checkout=str(checkout),dirty=dirty,existing_pixi=False,existing_venv=False,utc=datetime.now(timezone.utc).isoformat()))
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    python='.venv/c105-clean/Scripts/python.exe';runtime=['pixi','run','--locked',python]
    commands=[('pixi-install',['pixi','install','--locked']),
      ('venv',['pixi','run','--locked','python','-m','venv','--system-site-packages','.venv/c105-clean']),
      ('torch-install',runtime+['-m','pip','install','--ignore-installed','--require-hashes','--no-deps','-r','experiments/C1-03/requirements-win-cpu.lock']),
      ('runtime-install',runtime+['-m','pip','install','--ignore-installed','--require-hashes','--no-deps','-r','experiments/C1-03/stage2/runtime-supplement.lock']),
      ('pip-check',runtime+['-m','pip','check']),('artifact-audit',runtime+['scripts/audit_c103_artifacts.py']),
      ('witness',runtime+['scripts/c105_witness.py','verify','--witness',str(a.witness.resolve()),'--output',str(out/'witness-result.json')]),
      ('clean-status',['git','status','--porcelain'])]
    for label,argv in commands:
        folder=out/'commands'/label;folder.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
        result=subprocess.run(argv,cwd=checkout,env=env,capture_output=True)
        (folder/'stdout.log').write_bytes(result.stdout);(folder/'stderr.log').write_bytes(result.stderr)
        write(folder/'command.json',dict(argv=argv,cwd=str(checkout),start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
              stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
        print(label,result.returncode,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    write(out/'complete.json',dict(status='PASS',head=head,commands=len(commands),environment_copied=False,training_repeated='NOT_RUN'))


if __name__=='__main__':main()
