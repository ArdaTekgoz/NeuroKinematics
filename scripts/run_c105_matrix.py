"""Run the frozen matrix sequentially; fail immediately on a failed command."""
import json
from pathlib import Path
import subprocess
import sys
from neurokinematics.neural.c104 import read_json, write_json
from neurokinematics.neural.c105 import ROOT, STAGE2


def main():
    seeds=[2026100201,2026100202,2026100203]
    for experiment in ['E-C03','E-C04','E-C05']:
        if experiment=='E-C05':
            prior=[read_json(STAGE2/'E-C03'/f'seed-{seed}'/'attempt-001/summary.json') for seed in seeds]
            if not any(p['evaluation']['FK']['out_of_limits'] for p in prior):
                write_json(STAGE2/'E-C05-SKIP.json',dict(status='SKIP',reason='no E-C03 selected FK checkpoint has raw limit violations'))
                continue
        for seed in seeds:
            name=f's2-{experiment.lower()}-{seed}'
            command=[sys.executable,'scripts/c105_command.py','--name',name,'--','pixi','run','--locked',
                     '.venv/c103/Scripts/python.exe','scripts/run_c105.py','full','--experiment',experiment,'--seed',str(seed)]
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode: raise SystemExit(result.returncode)
    write_json(STAGE2/'matrix-complete.json',dict(status='COMPLETE',seeds=seeds,test_and_benchmark='SEALED_NOT_RUN'))


if __name__=='__main__': main()
