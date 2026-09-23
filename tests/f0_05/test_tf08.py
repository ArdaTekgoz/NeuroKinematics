"""T-F08 negative cases and deterministic easy targets, separate from full benchmark."""

from dataclasses import replace

import numpy as np
import pytest

from neurokinematics.benchmark.proof import prove_outer_reach,verify_outer_reach,classify_outer_target
from neurokinematics.benchmark.validation import CandidateValidator,deadline_success
from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.solvers.dls import DLS,SolverStatus


@pytest.mark.parametrize('joint',range(6))
def test_real_robot_easy_local_independent_pinocchio(robot,joint):
    q=np.array([.3,-.6,.8,-1.,.5,-.7]);goal=q.copy();goal[joint]+=.01
    pose=PinocchioFK(robot).reference_forward_kinematics(goal)
    quat=canonical_quaternion(pose[:3,:3]);p=pose[:3,3]
    result=DLS(robot).solve(q,p,quat)
    verified=CandidateValidator(robot).validate(result,p,quat)
    assert result.status==SolverStatus.SUCCESS
    assert verified.profile_a_geometry and verified.profile_b_geometry


@pytest.mark.parametrize('fault', ['wrong_shape','nan','inf','zero_quaternion','xyzw','outside_start'])
def test_bad_inputs_never_succeed(robot,fault):
    q=np.array([.3,-.6,.8,-1.,.5,-.7]);pose=PinocchioFK(robot).reference_forward_kinematics(q)
    p=pose[:3,3];quat=canonical_quaternion(pose[:3,:3])
    if fault=='wrong_shape':q=q[:5]
    if fault=='nan':p=p.copy();p[0]=np.nan
    if fault=='inf':p=p.copy();p[0]=np.inf
    if fault=='zero_quaternion':quat=np.zeros(4)
    if fault=='xyzw':quat=np.roll(quat,-1)
    if fault=='outside_start':q=q.copy();q[0]=999
    result=DLS(robot).solve(q,p,quat)
    if fault=='xyzw':
        assert result.status!=SolverStatus.SUCCESS or not CandidateValidator(robot).validate(result,p,canonical_quaternion(pose[:3,:3])).profile_b_geometry
    else:assert result.status==SolverStatus.INVALID_INPUT


def test_wrong_model_tcp_frame_order_rejected_by_identity(robot):
    q=np.array([.3,-.6,.8,-1.,.5,-.7])
    pose=PinocchioFK(robot).reference_forward_kinematics(q)
    for changed in (replace(robot,tcp='flange'),replace(robot,joint_names=robot.joint_names[::-1]),
                    replace(robot,base='absent')):
        try:
            wrong=CandidateValidator(changed).reference.reference_forward_kinematics(q)
        except ValueError:
            continue
        assert np.linalg.norm(wrong-pose)>1e-3


def test_analytic_outside_target_proof(robot):
    target=[100.,0.,0.]
    evidence=prove_outer_reach(target,robot)
    assert evidence is not None and verify_outer_reach(*evidence,robot)
    status,classified=classify_outer_target(target,robot)
    assert status==SolverStatus.PROVEN_UNREACHABLE and classified==evidence
    assert not verify_outer_reach({**evidence[0],'target_distance_m':99.},evidence[1],robot)
    assert prove_outer_reach([0.,0.,0.],robot) is None
    assert classify_outer_target([0.,0.,0.],robot)[0]==SolverStatus.UNRESOLVED
    q=np.zeros(6);result=DLS(robot).solve(q,target,[1,0,0,0],deadline_ns=1000000)
    assert result.status!=SolverStatus.PROVEN_UNREACHABLE


def test_timeout_and_late_success_never_deadline_pass(robot):
    q=np.zeros(6);pose=PinocchioFK(robot).reference_forward_kinematics(q)
    result=DLS(robot).solve(q,pose[:3,3],canonical_quaternion(pose[:3,:3]),deadline_ns=1)
    assert result.status==SolverStatus.TIMEOUT
    assert not deadline_success(True,result.status,1,10_000_000)
    assert not deadline_success(True,SolverStatus.SUCCESS,50_000_001,50_000_000)


def test_unresolved_not_proven(robot):
    q=np.zeros(6);pose=PinocchioFK(robot).reference_forward_kinematics(q)
    result=DLS(robot).solve(q,pose[:3,3],canonical_quaternion(pose[:3,:3]))
    altered=CandidateValidator(robot).validate(result,pose[:3,3]+[.2,0,0],canonical_quaternion(pose[:3,:3]))
    assert altered.status==SolverStatus.UNRESOLVED
    assert prove_outer_reach(pose[:3,3]+[.2,0,0],robot) is None
