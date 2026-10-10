"""T-C06: install fresh locked runtime, reproduce six models, run critical regressions."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
    checkout=a.checkout.resolve();out=a.output.resolve()
    if out.is_relative_to(checkout):raise ValueError('Evidence output must be outside clean checkout')
    if (checkout/'.pixi').exists() or (checkout/'.venv').exists():raise ValueError('fresh environment required')
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True)
    if dirty.strip():raise ValueError('clean checkout required')
    out.mkdir(parents=True,exist_ok=False)
    def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    write(out/'start.json',dict(head=head,checkout=str(checkout),dirty=dirty,existing_pixi=False,existing_venv=False,utc=datetime.now(timezone.utc).isoformat()))
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update({x:'1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    env.update(PYTHONUTF8='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    runtime=['pixi','run','--locked','.venv/c107-clean/Scripts/python.exe']
    commands=[('pixi-install',['pixi','install','--locked']),
        ('venv',['pixi','run','--locked','python','-m','venv','--system-site-packages','.venv/c107-clean']),
        ('torch-install',runtime+['-m','pip','install','--ignore-installed','--require-hashes','--no-deps','-r','experiments/C1-06R/requirements-win-cu128.lock']),
        ('pip-check',runtime+['-m','pip','check']),
        ('runtime',runtime+['-c',"import sys,torch,numpy,platform;from neurokinematics.neural import c107;print(sys.executable);print(c107.__file__);print(platform.platform());print(torch.__version__,numpy.__version__);print(torch.cuda.get_device_name(0))"]),
        ('witness',runtime+['scripts/c107_witness.py','verify','--witness','experiments/C1-07/closure/witness.json','--artifact-root',str(a.artifact_root.resolve()),'--output',str(out/'witness-result.json')])]
    suites={'handoff':['tests/c1_07'],
        'fk':['tests/c1_03/test_analytic.py','tests/c1_03/test_contract.py','tests/c1_03/test_mutations.py'],
        'physics':['tests/c1_05/test_physics.py'],
        'decoder':['tests/c1_06r/test_precision.py']}
    for name,paths in suites.items():commands.append((name,runtime+['-m','pytest',*paths,'-q','--junitxml='+str(out/(name+'.xml'))]))
    commands.append(('clean-status',['git','status','--porcelain']))
    for label,argv in commands:
        folder=out/'commands'/label;folder.mkdir(parents=True);start=datetime.now(timezone.utc).isoformat()
        result=subprocess.run(argv,cwd=checkout,env=env,capture_output=True)
        (folder/'stdout.log').write_bytes(result.stdout);(folder/'stderr.log').write_bytes(result.stderr)
        write(folder/'command.json',dict(argv=argv,cwd=str(checkout),start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
            stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
        print(label,result.returncode,flush=True)
        if result.returncode or (label=='clean-status' and result.stdout.strip()):raise SystemExit(result.returncode or 1)
    counts={}
    for name in suites:
        ts=ET.parse(out/(name+'.xml')).getroot().find('testsuite')
        counts[name]={k:int(ts.attrib[k]) for k in ['tests','failures','errors','skipped']}
        if any(counts[name][k] for k in ['failures','errors','skipped']):raise ValueError('Incomplete regression gate')
    write(out/'complete.json',dict(status='PASS',head=head,commands=len(commands),tests=counts,environment_copied=False,
        external_artifacts='six hash-checked checkpoints only',training_repeated='NOT_RUN',final_raw='NOT_READ',end_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
