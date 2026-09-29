"""Fresh C1-03 environment and full acceptance from a clean committed checkout.

Invoke with the host Python, in a new checkout. Nothing is copied from a prior
environment; package download caches may be reused. Every command is logged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path.cwd();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if (root/'.pixi').exists() or (root/'.venv/c103').exists():
        raise ValueError('fresh checkout must have no existing Pixi or Torch environment')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
    if dirty.strip(): raise ValueError('fresh checkout must be clean')
    (out/'clean-start.json').write_text(json.dumps({'head':head,'status':dirty,'checkout':str(root),'existing_pixi':False,'existing_venv':False},indent=2)+'\n',encoding='utf-8')
    counter=0
    def run(label,args):
        nonlocal counter
        counter+=1
        command=[sys.executable,'scripts/c103_command.py','--output',str(out),'--id',f'{counter:03d}-{label}','--',*args]
        subprocess.run(command,check=True)
    run('pixi-lock',['pixi','lock','--check'])
    run('pixi-install',['pixi','install','--locked'])
    run('venv',['pixi','run','--locked','python','-m','venv','--system-site-packages','.venv/c103'])
    runtime=['pixi','run','--locked','.venv/c103/Scripts/python.exe']
    run('torch-install',runtime+['-m','pip','install','--ignore-installed','--require-hashes','--no-deps','-r','experiments/C1-03/requirements-win-cpu.lock'])
    run('supplement',runtime+['-m','pip','install','--ignore-installed','--require-hashes','--no-deps','-r','experiments/C1-03/stage2/runtime-supplement.lock'])
    run('pip-check',runtime+['-m','pip','check'])
    run('artifact-audit',runtime+['scripts/audit_c103_artifacts.py'])
    run('analytic',runtime+['-m','pytest','-q','tests/c1_03/test_analytic.py','--junitxml='+str(out/'analytic.xml')])
    module=['-m','neurokinematics.core.torch_validation']
    run('smoke',runtime+module+['--smoke','--output',str(out/'smoke')])
    run('full',runtime+module+['--output',str(out/'full'),'--smoke-witness',str(out/'smoke')])
    run('unit-mutation',runtime+['-m','pytest','-q','tests/c1_03','--mutation-output='+str(out/'mutations'),'--junitxml='+str(out/'unit.xml')])
    for task in ['f0_01','f0_02','f0_03']:
        run(task,runtime+['-m','pytest','-q','tests/'+task,'--junitxml='+str(out/(task+'.xml'))])
    run('evidence-audit',runtime+module+['--output',str(out/'full'),'--verify'])
    (out/'complete.json').write_text(json.dumps({'status':'PASS','commands':counter,'head':head,'note':'independent fresh environment; no prior environment copied'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
