"""Create a fresh Windows worktree and log installation, reproduction and regression.

The overlay is only the reviewed F0-06 scripts, tests and explicit API parameters.
No environments, caches or generated data are copied from the source checkout.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_f06_regression import now, sha, write, run_commands

BASE = 'e7d211f42496f803688e2a510daca97e102092dc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path.cwd()
    worktree, output = args.worktree.resolve(), args.output.resolve()
    if worktree.exists() or output.exists():
        raise ValueError('worktree and evidence destination must both be new')
    output.mkdir(parents=True)
    records = []
    def run(name, argv, cwd):
        entry = {'name':name,'argv':argv,'command':subprocess.list2cmdline(argv),
                 'cwd':str(cwd),'started_utc':now()}
        result = subprocess.run(argv,cwd=cwd,env=dict(os.environ,PYTHONIOENCODING='utf-8'),capture_output=True)
        entry.update(finished_utc=now(),exit_code=result.returncode)
        (output/'logs').mkdir(exist_ok=True)
        for kind, value in [('stdout',result.stdout),('stderr',result.stderr)]:
            path = output/'logs'/f'{name}.{kind}.log'; path.write_bytes(value)
            entry[kind] = {'path':str(path),'sha256':sha(path)}
        records.append(entry);write(output/'setup-commands.json',records)
        print(name,result.returncode,flush=True)
        if result.returncode: raise RuntimeError(f'{name} failed; see logs')
        return result.stdout.decode('utf-8').strip()
    run('worktree',['git','worktree','add','--detach',str(worktree),BASE],source)
    head = run('head',['git','rev-parse','HEAD'],worktree)
    if head != BASE: raise ValueError('wrong clean worktree base')
    absent = {p: not (worktree/p).exists() for p in ('.pixi','.pytest_cache','data/generated','experiments/F0-06')}
    if not all(absent.values()): raise ValueError('fresh worktree contains reused outputs/cache')
    overlay = json.loads((source/'experiments/F0-06/overlay.json').read_bytes())
    for relative, expected in overlay['files'].items():
        if sha(source/relative) != expected: raise ValueError(f'overlay changed: {relative}')
        (worktree/relative).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,worktree/relative)
    pixi_version = run('pixi-version',['pixi','--version'],worktree)
    run('initial-install',['pixi','install','--locked'],worktree)
    run('lock-check',['pixi','lock','--check'],worktree)
    run('runtime',['pixi','run','--locked','python','-c',
        'import sys,platform,importlib.metadata as m; import numpy,pinocchio; print(sys.executable); print(platform.platform()); print(platform.machine()); print(sys.version); print({x:m.version(x) for x in ["pytest","hatchling","xacro","PyYAML"]}); print(numpy.__version__,pinocchio.__version__)'],worktree)
    write(output/'clean-environment.json',{'status':'PASS','base_commit':head,'worktree':str(worktree),
          'initial_paths_absent':absent,'pixi':pixi_version,'overlay':overlay,
          'global_download_cache':'USED; default shared Pixi cache was not cleared',
          'cache_free':False,'linux':'NOT_RUN','canonical_platform':'native Windows x64',
          'lock_sha256_before':sha(source/'pixi.lock'),'lock_sha256_after':sha(worktree/'pixi.lock')})
    run('reproduction-driver',[sys.executable,str(source/'scripts/run_f06_reproduction.py'),
        '--cwd',str(worktree),'--output',str(output)],source)
    run_commands(worktree,output/'regression')


if __name__ == '__main__':
    main()
