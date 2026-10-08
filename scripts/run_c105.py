"""Approved, gated C1-05 pilot and preregistered paired training CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from neurokinematics.neural import c105
from neurokinematics.neural.c104 import read_json, write_json, sha, load_data, reject_shifted_labels


def gate():
    subprocess.run([sys.executable,'scripts/check_c105_stage1.py','--check'],check=True)
    approval=read_json(c105.STAGE2/'approval.json')
    if approval['status']!='APPROVED' or approval['stage1_sha256sums']!=sha(c105.BASE/'SHA256SUMS'):
        raise ValueError('approval not bound to Stage1')
    domain=read_json(c105.STAGE2/'domain/summary.json')
    if domain['status']!='PASS': raise ValueError('domain gate')
    for path,expected in domain['sources'].items():
        if sha(c105.ROOT/path)!=expected: raise ValueError('domain source changed')
    for path in ['physics-tests.xml','training-tests.xml']:
        suites=ET.parse(c105.STAGE2/path).getroot().iter('testsuite')
        if any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')): raise ValueError('test gate: '+path)
    mutations=read_json(c105.STAGE2/'mutations-v2/summary.json')
    if mutations['status']!='PASS': raise ValueError('mutant gate')
    train,val=load_data(label_fk=True)
    negative=reject_shifted_labels(train.take(__import__('numpy').flatnonzero((train.mode=='local') & (train.family=='main'))[:64]))
    return negative


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['pilot','full'])
    p.add_argument('--experiment',choices=list(c105.PAIRS),default='E-C03')
    p.add_argument('--seed',type=int,default=2026100201); p.add_argument('--attempt',default='attempt-001')
    a=p.parse_args()
    if not a.attempt.replace('-','').isalnum(): p.error('safe attempt name required')
    negative=gate()
    if a.action=='full':
        pilot=read_json(c105.STAGE2/'pilot-assessment.json')
        if pilot['status']!='PASS_TO_FULL' or pilot['source_hashes']!=c105.source_hashes(): raise ValueError('pilot gate/source drift')
        existing=list(c105.STAGE2.glob('E-C*/seed-*/*/summary.json'))
        if any(read_json(x)['experiment']==a.experiment and read_json(x)['seed']==a.seed for x in existing): raise ValueError('completed seed already exists')
        if sum(2*read_json(x)['steps_per_model'] for x in existing)+6000>54000: raise ValueError('step budget')
        if sum(read_json(x)['wall_s'] for x in existing)>=36*3600: raise ValueError('wall budget')
        if a.experiment!='E-C03':
            ec03=[read_json(x) for x in existing if read_json(x)['experiment']=='E-C03']
            if sorted(x['seed'] for x in ec03)!=[2026100201,2026100202,2026100203]: raise ValueError('E-C03 three seeds prerequisite')
            if a.experiment=='E-C05' and not any(x['evaluation']['FK']['out_of_limits'] for x in ec03): raise ValueError('E-C05 condition is false; SKIP')
    result=c105.train_pair(a.experiment,a.seed,a.attempt,pilot=a.action=='pilot')
    if a.action=='pilot':
        # Full decision also reviewed against scalar/component/gradient logs before invoking full.
        assessment=dict(status='PASS_TO_FULL',source_hashes=c105.source_hashes(),negative_control=negative,
            pilot_summary=str(c105.STAGE2/'pilot'/f'seed-{a.seed}'/a.attempt/'summary.json'),
            q_fk_identity='all raw pilot rows preserved; no clamp',weights_retuned=False,
            domain_gate='PASS',pilot_epochs=result['epochs'],finite_losses_and_gradients=True)
        write_json(c105.STAGE2/'pilot-assessment.json',assessment)
    print(json.dumps({k:result[k] for k in ('status','experiment','seed','epochs','steps_per_model','wall_s','evaluation')}))


if __name__=='__main__': main()
