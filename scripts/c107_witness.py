"""Record/verify portable, stratified validation witness without opening test shards."""
import argparse,json,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from neurokinematics.neural import c107 as c


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['record','verify']);p.add_argument('--witness',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,default=c.ROOT);p.add_argument('--output',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((c.ROOT/'experiments/C1-07/preparation/handoff-manifest.json').read_text());c.verify_manifest(manifest)
    if a.action=='record':
        if a.witness.exists():raise FileExistsError(a.witness)
        from neurokinematics.neural.c104 import load_data
        _,val=load_data(label_fk=True);ids=[]
        for family in ['main','boundary','singularity']:
            for mode in ['local','wide']:ids.extend(np.flatnonzero((val.family==family)&(val.mode==mode))[:8].tolist())
        requests=[dict(pair_id=str(val.pair_id[i]),family=str(val.family[i]),mode=str(val.mode[i]),position_m=val.position[i].tolist(),quaternion_wxyz=val.quaternion[i].tolist(),q_current_rad=val.q_current[i].tolist()) for i in ids]
        records={x['id']:c.predict(x,requests,a.artifact_root) for x in manifest['candidates']}
        data=dict(schema='c107-witness-v1',manifest_sha256=c.sha(c.ROOT/'experiments/C1-07/preparation/handoff-manifest.json'),requests=requests,records=records,
            selection='first8 per family/mode from validation,48total; no success-based selection',batch_size=48,old_final_raw='NOT_READ')
        a.witness.parent.mkdir(parents=True,exist_ok=True);a.witness.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        print('RECORDED 48 requests x6 checkpoints; independent FK metrics')
    else:
        data=json.loads(a.witness.read_text());assert data['manifest_sha256']==c.sha(c.ROOT/'experiments/C1-07/preparation/handoff-manifest.json')
        for x in manifest['candidates']:
            measured=c.predict(x,data['requests'],a.artifact_root)
            assert measured==data['records'][x['id']],x['id']
        result=dict(status='PASS',checkpoints=6,requests=48,predictions=288,exact_q_and_independent_fk_metrics=True,
            witness_sha256=c.sha(a.witness),python=platform.python_version(),torch=str(torch.__version__),source_root=str(c.ROOT),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),training='NOT_RUN',final_raw='NOT_READ')
        if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result))


if __name__=='__main__':main()
