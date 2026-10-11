"""Post-hoc Core/Foundations sampling review; reads train shards only."""
from pathlib import Path
import numpy as np
from neurokinematics.neural import c104
from neurokinematics.neural import c106r_diagnostic5 as s
from neurokinematics.data import pairs


def main():
    d=s.d;output=s.BASE/'sampling-review.json'
    if output.exists():raise FileExistsError(output)
    config,_,norm=c104.contracts()
    train=c104._read_split('train',config,norm,label_fk=True)
    main=train.take(np.flatnonzero((train.family=='main')&(train.mode=='local')))
    rows=d.matched_rows(train,2048)[0]['local']
    robot=s.r.load_robot();bounds=np.asarray(robot.limits);span=bounds[:,1]-bounds[:,0]
    pcfg,_=pairs.load_contract()
    for i in range(len(main.pair_id)):
        source=dict(sample_id=str(main.source_sample_id[i]),q=main.q_target[i],group_id=str(main.group_id[i]),
                    family='main',split='train',position=main.position[i],quaternion=main.quaternion[i])
        generated=pairs.make_base(source,'local',pcfg,bounds)
        assert np.array_equal(generated['q_current'],main.q_current[i])
    assert len(set(main.source_sample_id))==len(main.pair_id)==7000
    keys={p:i for i,p in enumerate(main.source_sample_id)}
    database=(main.q_target-bounds[:,0])/span;queries=(rows.q_target-bounds[:,0])/span
    neighbors=[];distances=[]
    for start in range(0,len(rows.pair_id),128):
        value=queries[start:start+128]
        dist=((value[:,None,:]-database[None,:,:])**2).sum(-1)
        for k,source in enumerate(rows.source_sample_id[start:start+128]):dist[k,keys[source]]=np.inf
        choice=dist.argmin(1)
        neighbors.extend(choice.tolist());distances.extend(np.sqrt(dist[np.arange(len(choice)),choice]).tolist())
    correction=np.linalg.norm((rows.q_target-rows.q_current)/span,axis=1)
    ratios=np.asarray(distances)/correction
    raw=d.ROOT/'data/generated/C1-06R/diagnostic5/sampling-rows.json'
    if raw.exists():raise FileExistsError(raw)
    d.write_json(raw,dict(rows=[dict(pair_id=str(rows.pair_id[i]),neighbor_root=str(main.source_sample_id[neighbors[i]]),
        nearest_root_distance_normalized=distances[i],local_correction_distance_normalized=float(correction[i]),
        ratio=float(ratios[i])) for i in range(len(rows.pair_id))]))
    source_paths=[Path(__file__),d.ROOT/'experiments/F0-04/config.json',d.ROOT/'experiments/C1-02/config.json',
        d.ROOT/'experiments/F0-06/CORE_HANDOFF.md',Path(pairs.__file__),Path(c104.__file__)]
    d.write_json(output,dict(status='COMPLETE_POST_HOC_REVIEW',split='train_only',
        provenance_replayed=7000,main_train_roots=7000,local_pairs_per_root=1,
        diagnostic_roots=2048,eligible_matched_main_roots=int(((train.family=='main')&(train.mode=='wide')&train.label_present).sum()),
        nearest_search='2048 selected teacher q against7000 main train teacher q; self excluded; affine-limit Euclidean metric',
        nearest_root_distance_normalized=s.stats(distances),local_correction_distance_normalized=s.stats(correction),
        nearest_root_vs_correction_ratio=s.stats(ratios),
        nearest_root_selected_raw_joint_distance_rad=s.stats(np.linalg.norm(main.q_target[neighbors]-rows.q_target,axis=1)),
        joint_span_rad=span.tolist(),same_radian_error_Q_penalty_relative_to_joint2=(span[1]**2/span**2).tolist(),
        interpretation='One local perturbation per global root; density is descriptive, not proof of insufficient data or irreducible error. No labels generated or shards modified.',
        sources={str(p.relative_to(d.ROOT)):d.sha(p) for p in source_paths},
        raw=dict(path=str(raw.relative_to(d.ROOT)),sha256=d.sha(raw)),final_test='NOT_CREATED',old_final_raw='NOT_READ'))
    print('PASS:7000 train local provenance replays;2048 nearest-other-root measurements; no data modification')


if __name__=='__main__':main()
