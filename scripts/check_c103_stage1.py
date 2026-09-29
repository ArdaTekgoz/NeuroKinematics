"""C1-03 static contract/hash audit ONLY. Never evaluates FK or gradients."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.chain import extract_chain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/C1-03'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def samples(c, robot):
    lo, hi = np.array(robot.limits).T
    result=[]
    def add(key, group, q):
        q=np.asarray(q,dtype='<f8')
        q32=q.astype('<f4')
        q32=np.where(q32.astype('f8')<lo,np.nextafter(q32,np.float32(np.inf)),q32)
        q32=np.where(q32.astype('f8')>hi,np.nextafter(q32,np.float32(-np.inf)),q32).astype('<f4')
        assert np.all(q>=lo) and np.all(q<=hi)
        assert np.all(q32.astype('f8')>=lo) and np.all(q32.astype('f8')<=hi)
        result.append(dict(id=key,group=group,q64=q.tolist(),q32=q32.tolist(),
                           q64_sha256=digest(q.tobytes()),q32_sha256=digest(q32.tobytes())))
    margin=c['sampling']['margin_rad']
    for group,seed,count in [('fk',c['sampling']['fk_seed'],c['sampling']['fk_count']),
                             ('grad',c['sampling']['gradient_seed'],c['sampling']['gradient_count'])]:
        q=np.random.Generator(np.random.PCG64(seed)).uniform(lo+margin,hi-margin,size=(count,6))
        for i,row in enumerate(q): add(f'{group}-{i:04d}',group,row)
    mid=(lo+hi)/2
    add('hand-zero','hand',np.zeros(6))
    add('hand-midpoint','hand',mid)
    add('hand-mixed','hand',[.3,-.6,.8,-1.,.5,-.7])
    for i in range(6):
        for label,bound,direction in [('lower',lo,1),('upper',hi,-1)]:
            for mode,offset in [('exact',0),('near',5e-7)]:
                q=mid.copy(); q[i]=bound[i]+direction*offset
                add(f'hand-j{i+1}-{label}-{mode}','edge',q)
    for label,value in [('minus',-1e-8),('zero',0),('plus',1e-8)]:
        add(f'hand-wrist-{label}','singularity_candidate',[.3,-.6,.8,-1.,value,-.7])
    assert len(result)==c['sampling']['total_count']
    assert len({x['q64_sha256'] for x in result if x['group']=='grad'})==32
    return b''.join((json.dumps(x,separators=(',',':'),allow_nan=False)+'\n').encode() for x in result)

def audit():
    c=read('experiments/C1-03/config.json')
    robot=load_robot(ROOT)
    assert c['joint_names']==list(robot.joint_names)
    assert (c['robot_id'],c['base'],c['tcp'])==(robot.robot_id,robot.base,robot.tcp)
    assert c['fk']=={'float64':{'position_l2_m':1e-9,'rotation_frobenius':1e-9},'float32':{'position_l2_m':1e-5,'rotation_frobenius':1e-5},'aggregate':'every sample passes both; no averages gate','homogeneous_atol':{'float64':1e-9,'float32':1e-5}}
    assert c['gradient']['epsilon_rad']==1e-6 and c['gradient']['atol']==1e-5 and c['gradient']['rtol']==1e-3
    assert c['tests']=={'T-C01':'NOT_RUN','T-C02':'NOT_RUN'}
    chain=extract_chain(robot.urdf,robot.base,robot.tcp,robot.joint_names)
    assert [j.name for j in chain]==list(robot.joint_names)+['joint_6-flange','flange-tool0']
    for j in chain:
        if j.kind=='revolute': assert j.limits==robot.limits[robot.joint_names.index(j.name)]
    frozen=read('experiments/F0-06/handoff-inputs.json')
    for path,expected in frozen.items():
        assert digest((ROOT/path).read_bytes())==expected, path
    manifest=read('assets/robots/robot_a/manifest.json')
    for entry in manifest['files']:
        assert digest((ROOT/entry['path']).read_bytes())==entry['sha256'],entry['path']
    pins=read('experiments/C1-03/dependency-pins.json')
    assert next(p['version'] for p in pins if p['name']=='torch')=='2.10.0+cpu'
    lock=(OUT/'requirements-win-cpu.lock').read_text(encoding='utf-8')
    for p in pins: assert p['url']+' --hash=sha256:'+p['sha256'] in lock
    # Shared parser/NumPy FK are not additional independent references.
    paths=set(frozen)|{e['path'] for e in manifest['files']}
    paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/neurokinematics/kinematics').glob('*.py'))
    for folder in ['tests/f0_01','tests/f0_02','tests/f0_03','tests/c1_02']:
        paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/folder).glob('*.py'))
    paths.update(['pixi.toml','pyproject.toml','experiments/F0-06/G0_DECISION.md','experiments/F0-06/CORE_HANDOFF.md','experiments/F0-06/handoff-inputs.json','experiments/F0-02/config.json','experiments/F0-02/fk-validation-summary.json','experiments/F0-03/config.json','experiments/F0-03/acceptance.json','experiments/C1-02/acceptance.json','experiments/C1-02/config.json','experiments/C1-02/schema.json','src/neurokinematics/data/pairs.py','src/neurokinematics/data/pair_validation.py','docs/raporlar/02_Core_v1_0_r1.md','docs/TEST_PROTOCOL.md'])
    entries=[]
    for path in sorted(paths):
        data=(ROOT/path).read_bytes()
        blob=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
        # Store distinct raw and Git blob SHA256; neither is a Git object id.
        entries.append(dict(path=path,working_tree_sha256=digest(data),git_blob_content_sha256=digest(blob),canonical_lf_sha256=digest(data.replace(b'\r\n',b'\n'))))
    return c, samples(c,robot), {'schema_version':1,'baseline_commit':'b82faf95ef87666471fd5ae77ec1d392364a108a','hash_policy':'raw bytes required for frozen assets; canonical LF only for explicitly textual contract/evidence; raw and git blob SHA256 distinguished','files':entries}, {'handoff_files':len(frozen),'robot_manifest_files':len(manifest['files']),'input_files':len(entries),'chain':[{'name':j.name,'kind':j.kind,'origin':j.origin.tolist(),'axis':None if j.axis is None else j.axis.tolist(),'limits':j.limits} for j in chain]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--check',action='store_true');args=p.parse_args()
    c,sample_bytes,manifest,details=audit()
    if args.freeze:
        for name,data in [('samples.jsonl',sample_bytes),('input-hashes.json',encoded(manifest))]:
            target=OUT/name
            if target.exists(): raise ValueError('refusing to overwrite frozen '+str(target))
            target.write_bytes(data)
    assert (OUT/'samples.jsonl').read_bytes()==sample_bytes
    previous=read('experiments/C1-03/input-hashes.json')
    # Historical raw checkout endings may differ; immutable G0 inputs were checked raw above.
    for old,new in zip(previous['files'],manifest['files'],strict=True):
        assert old['path']==new['path'] and old['canonical_lf_sha256']==new['canonical_lf_sha256'],new['path']
        assert old['git_blob_content_sha256']==new['git_blob_content_sha256'],new['path']
    if (OUT/'SHA256SUMS').exists():
        for line in (OUT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            expected,path=line.split('  ',1)
            assert digest((ROOT/path).read_bytes().replace(b'\r\n',b'\n'))==expected,path
    report={'status':'PASS_STATIC_ONLY','T-C01':'NOT_RUN','T-C02':'NOT_RUN','sample_count':1086,'gradient_configurations':32,'samples_sha256':digest(sample_bytes),'config_sha256':digest((OUT/'config.json').read_bytes()),**details}
    if args.freeze:
        (OUT/'stage1-check.json').write_bytes(encoded(report))
        env={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'packages':{x:metadata.version(x) for x in ['numpy','pin','pytest']},'torch_installed':any(d.metadata['Name'].lower()=='torch' for d in metadata.distributions()),'pixi':subprocess.check_output(['pixi','--version'],text=True).strip(),'torch_selected':'2.10.0+cpu','torch_runtime':'NOT_RUN','gpu':'NOT_RUN','RAM':'NOT_MEASURED'}
        (OUT/'environment.json').write_bytes(encoded(env))
    print(json.dumps({k:v for k,v in report.items() if k!='chain'},indent=2))

if __name__=='__main__': main()
