"""Close C1-03 only after both complete measured runs and clean reproduction."""
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from neurokinematics.core.torch_validation import (ROOT, load_contract, source_hashes,
    sha, verify_evidence, write_json)

OUT=ROOT/'experiments/C1-03/stage2'

def junit(path,expected):
    tree=ET.parse(path)
    cases=tree.findall('.//testcase')
    assert len(cases)==expected,(path,len(cases))
    assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped'),path
    return dict(tests=len(cases),failures=0,errors=0,skipped=0)

def main():
    config,rows=load_contract()
    folders=[OUT/'full-exact-a',OUT/'clean-b/full']
    summaries=[verify_evidence(p,config,rows) for p in folders]
    environments=[json.loads((p/'environment.json').read_text()) for p in folders]
    assert summaries[0]['source_hashes']==summaries[1]['source_hashes']==source_hashes()
    for key in ['packages','config_sha256','samples_sha256']:
        assert environments[0][key]==environments[1][key],key
    for path in [OUT/'smoke-exact-a',OUT/'clean-b/smoke']:
        verify_evidence(path,config,rows,smoke=True)
    completed=json.loads((OUT/'clean-b/complete.json').read_text())
    start=json.loads((OUT/'clean-b/clean-start.json').read_text())
    assert completed['status']=='PASS' and completed['commands']==15
    assert start['status']=='' and not start['existing_pixi'] and not start['existing_venv']
    assert start['head']==completed['head']
    for cmd in (OUT/'clean-b/commands').glob('*/command.json'):
        assert json.loads(cmd.read_text())['exit_code']==0,cmd
    # Both artifact audits establish exact URL and SHA, not only versions.
    artifacts=[]
    for path in [OUT/'commands/024-artifact-audit/stdout.log',OUT/'clean-b/commands/007-artifact-audit/stdout.log']:
        audit=json.loads(path.read_text(encoding='utf-8'))
        assert audit['status']=='PASS' and len(audit['artifacts'])==11
        artifacts.append({v['name']:(v['version'],v['sha256']) for v in audit['artifacts']})
    assert artifacts[0]==artifacts[1]
    tests={}
    for path,count in [('unit-exact-a.xml',110),('f01-exact-a.xml',16),('f02-exact-a.xml',102),('f03-exact-a.xml',159),
                       ('clean-b/analytic.xml',19),('clean-b/unit.xml',110),('clean-b/f0_01.xml',16),('clean-b/f0_02.xml',102),('clean-b/f0_03.xml',159)]:
        tests[path]=junit(OUT/path,count)
    mutations=[]
    for path in [OUT/'mutations-exact-a',OUT/'clean-b/mutations']:
        records=[json.loads(p.read_text()) for p in path.glob('*.json')]
        assert len(records)==24 and all(r['status']=='KILLED' and r['baseline']=='PASS' for r in records)
        mutations.append({r['id']:r['mutant_sha256'] for r in records})
    assert mutations[0]==mutations[1]
    a=(folders[0]/'results.jsonl').read_bytes();b=(folders[1]/'results.jsonl').read_bytes()
    raw=[json.loads(line) for line in a.decode().splitlines()]
    worst={}
    for dtype in ['float64','float32']:
        group=[r for r in raw if r['kind']=='fk-'+dtype]
        worst[dtype]={metric:{'id':max(group,key=lambda r:r[metric])['id'],'value':max(r[metric] for r in group)} for metric in ['position_l2_m','rotation_frobenius']}
    g=max((r for r in raw if r['kind']=='gradient'),key=lambda r:r['max_abs'])
    delta=np.abs(np.array(g['autograd'])-np.array(g['finite_difference']))
    i,j=np.unravel_index(np.argmax(delta),delta.shape)
    grad=dict(id=g['id'],component=int(i),joint_index=int(j),autograd=g['autograd'][i][j],finite_difference=g['finite_difference'][i][j],max_abs=g['max_abs'],epsilon=g['epsilon'],configuration_count=32,comparison_count=2880,**{k:summaries[0]['gradient'][k] for k in ['max_relative','max_tolerance_ratio']})
    result={'status':'PASS / ACCEPTED','task':'C1-03','requirement':'REQ-C02','tests':{'T-C01':'PASS','T-C02':'PASS'},
            'protocol':'stage1 math r1 + stage2 runtime/harness r2','source_hashes':source_hashes(),
            'config_sha256':environments[0]['config_sha256'],'samples_sha256':environments[0]['samples_sha256'],
            'worst_fk':worst,'gradient':grad,'numerical_records_per_run':len(raw),'counts':summaries[0]['counts'],
            'source_mutants_per_run':24,'mutation_status':'24 KILLED / 0 SURVIVED / 0 ERROR',
            'regression_count_per_run':277,'c102_interface_tests_in_unit':21,'junit':tests,
            'clean_reproduction':completed,'result_bytes_identical':a==b,
            'raw_evidence':[{'path':p.relative_to(ROOT).as_posix()+'/results.jsonl','sha256':sha((p/'results.jsonl').read_bytes()),'bytes':(p/'results.jsonl').stat().st_size,'rows':len(raw),'storage':'GIT_TRACKED'} for p in folders],
            'limitations':{'Linux':'NOT_RUN','CUDA':'NOT_RUN','performance':'NOT_MEASURED','human_effort':'NOT_MEASURED','physical_robot':'NOT_RUN','collision_safety':'NOT_CHECKED','C1-04':'NOT_STARTED'}}
    write_json(OUT/'acceptance.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
