"""Portable checkpoint inference witness; clean verification never opens dataset shards."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from neurokinematics.neural import c105
from neurokinematics.neural.c104 import load_data, read_json, write_json, sha


def compute(checkpoint,features):
    import numpy as np
    model,metadata=c105.load_checkpoint(Path(checkpoint['path']))
    q,z=c105.infer(model,metadata['variant'],np.asarray(features,dtype=np.float32))
    inputs=c105.load_robot(); fk=c105.PinocchioFK(inputs); lo,hi=np.asarray(inputs.limits).T
    poses=[]
    for value in q:
        poses.append(fk.reference_forward_kinematics(value).tolist() if np.all(value>=lo) and np.all(value<=hi) else None)
    return dict(q_rad=q.tolist(),poses=poses)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['record','verify']);p.add_argument('--witness',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    if a.action=='record':
        _,val=load_data(label_fk=False)
        selected=val.take(__import__('numpy').arange(10))
        records=[]
        for path in sorted(c105.STAGE2.glob('E-C*/seed-*/*/summary.json')):
            summary=read_json(path)
            for variant,cp in summary['best_checkpoints'].items():
                records.append(dict(experiment=summary['experiment'],seed=summary['seed'],variant=variant,checkpoint=cp,
                                    **compute(cp,selected.conditioned.tolist())))
        frozen={str(path.relative_to(c105.ROOT)).replace('\\','/'):sha(path) for path in [c105.CONFIG,c105.BASE/'input-hashes.json',c105.ROOT/'experiments/C1-02/normalization.json']}
        frozen.update(c105.source_hashes());frozen.update(c105.load_robot().hashes)
        write_json(a.witness,dict(schema='c105-witness-v1',frozen=frozen,pair_ids=selected.pair_id.tolist(),features=selected.conditioned.tolist(),records=records,test_and_benchmark='SEALED_NOT_RUN'))
        print(json.dumps(dict(status='RECORDED',checkpoints=len(records),samples_each=10)))
    else:
        witness=read_json(a.witness)
        for rel,value in witness['frozen'].items():
            if sha(c105.ROOT/rel)!=value:raise ValueError('frozen drift: '+rel)
        for record in witness['records']:
            cp=record['checkpoint']; path=Path(cp['path'])
            if path.stat().st_size!=cp['bytes'] or sha(path)!=cp['sha256']: raise ValueError('weight SHA drift')
            result=compute(cp,witness['features'])
            if result['q_rad']!=record['q_rad'] or result['poses']!=record['poses']: raise ValueError('inference/FK witness drift')
        result=dict(status='PASS',checkpoints=len(witness['records']),samples_each=10,max_q_abs_rad=0,max_fk_element_abs=0,
                    frozen_files=len(witness['frozen']),witness_sha256=sha(a.witness),source_root=str(c105.ROOT),
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c105.ROOT,text=True).strip(),
                    test_and_benchmark='SEALED_NOT_RUN',training_reproduction='NOT_RUN',weights='LOCAL_ONLY')
        if a.output:write_json(a.output,result)
        print(json.dumps(result))


if __name__=='__main__':main()
