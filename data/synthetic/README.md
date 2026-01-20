# Synthetic Dataset — KUKA KR6 (A-4.2)

This dataset is synthetically generated based on the kinematic structure of a
KUKA KR6–type 6-DOF industrial robotic manipulator.

## Robot Model
- Robot: KUKA KR6 (6-DOF)
- Kinematic model: Denavit–Hartenberg (DH)
- Joint type: Revolute (R-R-R-R-R-R)
- Workspace: Industrial pick-and-place / manipulation range

## Dataset Generation
- Joint angles are uniformly sampled within the physical joint limits of KR6
- Forward kinematics is computed analytically using DH parameters
- End-effector Cartesian position (x, y, z) is recorded
- No sensor noise or model perturbation is applied (noise-free FK)

## Units
- Joint angles: radians
- End-effector position: meters

## Purpose
This dataset is used to validate the inverse kinematics learning pipeline
under forward-kinematics consistency constraints (A-4.2 stage),
before introducing adaptive loss weighting and hard-sample mining.

## Notes
- This dataset is fully synthetic and does not contain real robot measurements
- The dataset is reproducible via the provided data generation script
- The goal is methodological validation, not real-world deployment
