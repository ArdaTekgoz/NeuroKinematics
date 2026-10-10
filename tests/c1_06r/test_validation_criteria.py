from dataclasses import replace
import math
import numpy as np
from neurokinematics.neural import c106r_feature_scale as z


def test_profile_thresholds_use_metres_degrees_and_conjunction():
    roots=z.load_source('directions').take(np.arange(8))
    q=roots.q_target.copy();fk=z.f.s.IndependentFK(z.r.load_robot())
    distances=[.001999,.002001,0,0,.000999,.001001,0,0]
    angles=[0,0,.999,1.001,0,0,.499,.501]
    positions=[];rotations=[]
    for i in range(8):
        t=fk.forward_kinematics(q[i]);a=math.radians(angles[i])
        rz=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
        positions.append(t[:3,3]+[distances[i],0,0]);rotations.append(z.r.canonical_quaternion(rz@t[:3,:3]))
    rows=replace(roots,position=np.array(positions),quaternion=np.array(rotations))
    m=z.d.geometric_metrics(q,rows,details=True)
    assert [x['profile_a'] for x in m['rows']]==[True,False,True,False,True,True,True,True]
    assert [x['profile_b'] for x in m['rows']]==[False,False,False,False,True,False,True,False]


def test_invalid_and_missing_labels_remain_in_denominator():
    roots=z.load_source('directions').take(np.arange(4));q=roots.q_target.copy()
    q[1,0]=np.nan;q[2,0]=np.asarray(z.r.load_robot().limits)[0,1]+.001
    rows=replace(roots,label_present=np.zeros(4,dtype=bool),q_target=np.full_like(q,np.nan))
    m=z.d.geometric_metrics(q,rows,details=True)
    assert m['n']==4 and m['profile_a']==m['profile_b']==2
    assert m['nonfinite']==1 and m['out_of_limits']==1


def test_joint_label_is_not_the_success_metric_and_quaternion_sign_is_irrelevant():
    rows=z.load_source('directions').take(np.arange(8));q=rows.q_target.copy()
    expected=z.d.geometric_metrics(q,rows,details=True)
    rows=replace(rows,q_target=np.full_like(q,42),quaternion=-rows.quaternion)
    assert z.d.geometric_metrics(q,rows,details=True)==expected
