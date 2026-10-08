"""Read-only audit of the committed evidence snapshot; no data split is decoded."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args()
    root=a.root.resolve();base=root/'experiments/C1-05/stage2'
    manifest=json.loads((base/'evidence-manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path=(root/record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError('evidence drift: '+record['path'])
    audit=json.loads((base/'results-audit.json').read_text(encoding='utf-8'))
    if audit['status']!='PASS_EVIDENCE_AUDIT' or audit['totals']['model_seed_runs']!=18 or audit['totals']['validation_rows']!=64800:
        raise ValueError('acceptance evidence incomplete')
    for run in audit['runs']:
        for name,sha in run['source_hashes'].items():
            if digest(root/name)!=sha:raise ValueError('checkpoint source drift: '+name)
    clean=json.loads((base/'clean/witness-result.json').read_text())
    if clean['status']!='PASS' or clean['checkpoints']!=18 or clean['samples_each']!=10 or clean['max_q_abs_rad']!=0 or clean['max_fk_element_abs']!=0:
        raise ValueError('clean witness incomplete')
    if (base/'clean/commands/clean-status/stdout.log').read_bytes().strip():
        raise ValueError('witness checkout was not clean')
    for cp in audit['checkpoints']:
        path=Path(cp['path'])
        if not path.is_file() or path.stat().st_size!=cp['bytes'] or digest(path)!=cp['sha256']:raise ValueError('LOCAL_ONLY checkpoint inaccessible or drifted')
    commands=0
    for path in list((root/'experiments/C1-05/commands').glob('s2-*/command.json'))+list((base/'clean/commands').glob('*/command.json')):
        record=json.loads(path.read_text())
        expected_exit=1 if path.parent.name=='s2-mutants' else 0
        if record['exit_code']!=expected_exit:raise ValueError('unexpected command exit: '+str(path))
        for kind in ('stdout','stderr'):
            if digest(path.parent/(kind+'.log'))!=record[kind+'_sha256']:raise ValueError('command bytes drift')
        commands+=1
    result=dict(status='PASS',files=len(manifest['files']),checkpoints=len(audit['checkpoints']),commands=commands,
                root=str(root),manifest_sha256=digest(base/'evidence-manifest.json'),test_and_benchmark='SEALED_NOT_RUN')
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
