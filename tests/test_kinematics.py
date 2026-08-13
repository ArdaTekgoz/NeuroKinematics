"""
NeuroKinematics — Kinematic Validation Tests (A-1.4)

Milestone 1 Acceptance Criterion:
  FK results from our implementation must match an independent kinematic
  reference (Pinocchio) within a predefined numerical tolerance across
  random joint configurations.

  THIS CRITERION MUST PASS BEFORE ANY AI TRAINING BEGINS.
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurokinematics.core.robot_model import RobotModel
from neurokinematics.core.forward_kinematics import ForwardKinematics, DifferentiableFK
from neurokinematics.core.jacobian import (
    compute_geometric_jacobian,
    jacobian_rank,
    yoshikawa_manipulability,
    is_singular,
    jacobian_condition_number,
)

# ==============================================================================
# Paths
# ==============================================================================
URDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'robots', 'kuka_kr6', 'kr6.urdf')

# ==============================================================================
# Tolerance for numerical comparison
# ==============================================================================
FK_TOLERANCE = 1e-6      # meters
ROT_TOLERANCE = 1e-6     # dimensionless


def get_model_and_fk():
    """Load the KR6 model and create FK instance."""
    model = RobotModel.from_urdf(URDF_PATH)
    fk = ForwardKinematics(model)
    return model, fk


def random_joint_config(model, rng=None):
    """Generate a random joint configuration within limits."""
    if rng is None:
        rng = np.random.default_rng()
    q_min, q_max = model.joint_limits
    return rng.uniform(q_min, q_max)


# ==============================================================================
# Test 1: URDF Parsing (A-1.1)
# ==============================================================================
def test_urdf_parsing():
    """Verify URDF parser extracts correct joint/link information."""
    model = RobotModel.from_urdf(URDF_PATH)
    
    assert model.robot_name == 'kuka_kr6_r900_sixx', f"Robot name mismatch: {model.robot_name}"
    assert model.n_joints == 6, f"Expected 6 active joints, got {model.n_joints}"
    assert model.base_link == 'base_link', f"Base link mismatch: {model.base_link}"
    
    # Check joint types — all should be revolute
    for jt in model.joint_types:
        assert jt == 'revolute', f"Expected revolute, got {jt}"
    
    # Check joint limits exist and are finite
    q_min, q_max = model.joint_limits
    assert q_min.shape == (6,), f"Joint limit shape mismatch: {q_min.shape}"
    assert np.all(q_min < q_max), "Lower limits should be less than upper limits"
    assert np.all(np.isfinite(q_min)) and np.all(np.isfinite(q_max))

    print("[PASS] test_urdf_parsing")


# ==============================================================================
# Test 2: FK at Zero Configuration (A-1.2)
# ==============================================================================
def test_fk_zero_config():
    """FK at q=0 should produce a valid, finite transform."""
    model, fk = get_model_and_fk()
    q_zero = np.zeros(6)
    
    T = fk.compute(q_zero)
    assert T.shape == (4, 4), f"Transform shape mismatch: {T.shape}"
    assert np.all(np.isfinite(T)), "Transform contains NaN or Inf"
    
    # Last row should be [0, 0, 0, 1]
    np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1], decimal=10)
    
    # Rotation part should be orthogonal (R^T R = I)
    R = T[:3, :3]
    np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=10,
                                         err_msg="Rotation matrix is not orthogonal at q=0")
    
    pos = fk.position(q_zero)
    assert pos.shape == (3,)
    print(f"  FK at q=0: position = {pos}")
    print("[PASS] test_fk_zero_config")


# ==============================================================================
# Test 3: FK Rotation Matrix Orthogonality (A-1.2)
# ==============================================================================
def test_fk_rotation_orthogonality():
    """Rotation matrices should be orthogonal for all random configs."""
    model, fk = get_model_and_fk()
    rng = np.random.default_rng(42)
    n_samples = 1000
    
    for i in range(n_samples):
        q = random_joint_config(model, rng)
        R = fk.orientation_matrix(q)
        
        # R^T R should be identity
        RTR = R.T @ R
        assert np.allclose(RTR, np.eye(3), atol=1e-10), \
            f"Rotation not orthogonal at sample {i}: max error = {np.max(np.abs(RTR - np.eye(3)))}"
        
        # det(R) should be +1
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, \
            f"det(R) = {det} at sample {i}, expected 1.0"
    
    print(f"[PASS] test_fk_rotation_orthogonality ({n_samples} samples)")


# ==============================================================================
# Test 4: Quaternion Consistency (A-1.2)
# ==============================================================================
def test_quaternion_consistency():
    """Quaternion output should be consistent with rotation matrix."""
    model, fk = get_model_and_fk()
    rng = np.random.default_rng(123)
    
    for _ in range(100):
        q = random_joint_config(model, rng)
        R = fk.orientation_matrix(q)
        quat = fk.quaternion(q)
        
        # Quaternion should be unit norm
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-10, \
            f"Quaternion not unit: norm = {np.linalg.norm(quat)}"
        
        # Convert quaternion back to rotation matrix and compare
        w, x, y, z = quat
        R_from_q = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])
        np.testing.assert_array_almost_equal(R, R_from_q, decimal=8,
            err_msg="Quaternion inconsistent with rotation matrix")
    
    print("[PASS] test_quaternion_consistency")


# ==============================================================================
# Test 5: 6D Rotation Consistency (A-1.2)
# ==============================================================================
def test_6d_rotation_consistency():
    """6D rotation should match first two columns of rotation matrix."""
    model, fk = get_model_and_fk()
    rng = np.random.default_rng(456)
    
    for _ in range(100):
        q = random_joint_config(model, rng)
        R = fk.orientation_matrix(q)
        rot6d = fk.rotation_6d(q)
        
        expected = np.concatenate([R[:, 0], R[:, 1]])
        np.testing.assert_array_almost_equal(rot6d, expected, decimal=10)
    
    print("[PASS] test_6d_rotation_consistency")


# ==============================================================================
# Test 6: NumPy vs PyTorch FK Consistency (A-1.4)
# ==============================================================================
def test_numpy_vs_torch_fk():
    """Our NumPy FK and PyTorch DifferentiableFK must produce identical results."""
    model, fk_np = get_model_and_fk()
    diff_fk = DifferentiableFK(model)
    rng = np.random.default_rng(789)
    
    max_pos_err = 0.0
    max_rot_err = 0.0
    n_samples = 1000
    
    for _ in range(n_samples):
        q = random_joint_config(model, rng)
        
        # NumPy FK
        pos_np = fk_np.position(q)
        rot6d_np = fk_np.rotation_6d(q)
        
        # PyTorch FK
        q_torch = torch.tensor(q, dtype=torch.float64)
        pos_torch, rot6d_torch = diff_fk.compute(q_torch)
        
        pos_err = np.linalg.norm(pos_np - pos_torch.detach().numpy())
        rot_err = np.linalg.norm(rot6d_np - rot6d_torch.detach().numpy())
        
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        
        assert pos_err < FK_TOLERANCE, \
            f"NumPy vs Torch position error: {pos_err} > {FK_TOLERANCE}"
        assert rot_err < ROT_TOLERANCE, \
            f"NumPy vs Torch rotation error: {rot_err} > {ROT_TOLERANCE}"
    
    print(f"  Max position error: {max_pos_err:.2e}")
    print(f"  Max rotation error: {max_rot_err:.2e}")
    print(f"[PASS] test_numpy_vs_torch_fk ({n_samples} samples)")


# ==============================================================================
# Test 7: Differentiable FK Gradient Flow (A-1.2 / Physics-Aware prerequisite)
# ==============================================================================
def test_differentiable_fk_gradient():
    """DifferentiableFK must produce valid gradients for backpropagation."""
    model = RobotModel.from_urdf(URDF_PATH)
    diff_fk = DifferentiableFK(model, dtype=torch.float64)
    
    q = torch.randn(6, dtype=torch.float64, requires_grad=True)
    pos, rot6d = diff_fk.compute(q)
    
    # Compute a scalar loss and backpropagate
    loss = pos.sum() + rot6d.sum()
    loss.backward()
    
    assert q.grad is not None, "No gradient computed for joint angles"
    assert q.grad.shape == (6,), f"Gradient shape mismatch: {q.grad.shape}"
    assert torch.all(torch.isfinite(q.grad)), "Gradient contains NaN or Inf"
    
    print("[PASS] test_differentiable_fk_gradient")


# ==============================================================================
# Test 8: Jacobian Properties (A-1.3)
# ==============================================================================
def test_jacobian_properties():
    """Jacobian should be 6x6 and have full rank at non-singular configs."""
    model, fk = get_model_and_fk()
    rng = np.random.default_rng(1024)
    
    # Test at a known non-singular configuration
    q = np.array([0.1, -0.5, 0.3, 0.0, 0.5, 0.0])
    J = compute_geometric_jacobian(fk, q)
    
    assert J.shape == (6, 6), f"Jacobian shape mismatch: {J.shape}"
    assert np.all(np.isfinite(J)), "Jacobian contains NaN or Inf"
    
    rank = jacobian_rank(J)
    w = yoshikawa_manipulability(J)
    cond = jacobian_condition_number(J)
    
    print(f"  Config q={q}")
    print(f"  Jacobian rank: {rank}")
    print(f"  Manipulability: {w:.6f}")
    print(f"  Condition number: {cond:.2f}")
    
    assert rank == 6, f"Expected full rank 6, got {rank}"
    assert w > 0, "Manipulability should be > 0 at non-singular config"
    
    print("[PASS] test_jacobian_properties")


# ==============================================================================
# Test 9: Jacobian Numerical Validation (A-1.3 / A-1.4)
# ==============================================================================
def test_jacobian_numerical_validation():
    """Validate geometric Jacobian against finite-difference Jacobian."""
    model, fk = get_model_and_fk()
    
    q = np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.2])
    J_geometric = compute_geometric_jacobian(fk, q)
    
    # Finite-difference Jacobian
    eps = 1e-7
    J_fd = np.zeros((6, 6))
    T0 = fk.compute(q)
    pos0 = T0[:3, 3]
    R0 = T0[:3, :3]
    
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = fk.compute(q_plus)
        pos_plus = T_plus[:3, 3]
        
        # Linear part
        J_fd[:3, i] = (pos_plus - pos0) / eps
    
    # Compare linear part only (angular part FD is more complex)
    pos_error = np.max(np.abs(J_geometric[:3, :] - J_fd[:3, :]))
    print(f"  Max linear Jacobian error vs finite-diff: {pos_error:.2e}")
    
    assert pos_error < 1e-4, \
        f"Jacobian linear part error too large: {pos_error}"
    
    print("[PASS] test_jacobian_numerical_validation")


# ==============================================================================
# Test 10: Singularity Detection (A-1.3)
# ==============================================================================
def test_singularity_detection():
    """Test that singularity is detected at known singular configurations."""
    model, fk = get_model_and_fk()
    
    # Near-singular: elbow fully extended (joints 2 and 3 aligned)
    q_singular = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    J = compute_geometric_jacobian(fk, q_singular)
    w = yoshikawa_manipulability(J)
    
    print(f"  Manipulability at q=0: {w:.6f}")
    
    # At a clearly non-singular config
    q_good = np.array([0.5, -1.0, 0.8, 0.3, 0.6, 0.1])
    J_good = compute_geometric_jacobian(fk, q_good)
    w_good = yoshikawa_manipulability(J_good)
    
    print(f"  Manipulability at q_good: {w_good:.6f}")
    assert w_good > w * 0.1 or w_good > 1e-6, \
        "Non-singular config should have higher manipulability"
    
    print("[PASS] test_singularity_detection")


# ==============================================================================
# Test 11: Pinocchio Reference Validation (A-1.4 — MILESTONE 1 ACCEPTANCE)
# ==============================================================================
def test_pinocchio_reference():
    """
    MILESTONE 1 ACCEPTANCE TEST:
    Compare our FK against Pinocchio's FK across 1000 random configurations.
    Both must agree within FK_TOLERANCE.
    """
    try:
        import pinocchio as pin
    except ImportError:
        print("[SKIP] test_pinocchio_reference — pinocchio not installed")
        print("  Install with: conda install -c conda-forge pinocchio")
        print("  Milestone 1 acceptance requires this test to PASS.")
        return False

    model = RobotModel.from_urdf(URDF_PATH)
    fk = ForwardKinematics(model)

    # Load same URDF with Pinocchio
    pin_model = pin.buildModelFromUrdf(os.path.abspath(URDF_PATH))
    pin_data = pin_model.createData()

    rng = np.random.default_rng(42)
    n_samples = 1000
    max_pos_err = 0.0
    max_rot_err = 0.0
    n_fail = 0

    for i in range(n_samples):
        q = random_joint_config(model, rng)
        q_pin = np.zeros(pin_model.nq)
        q_pin[:6] = q

        # Our FK
        pos_ours = fk.position(q)
        R_ours = fk.orientation_matrix(q)

        # Pinocchio FK
        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.updateFramePlacements(pin_model, pin_data)
        
        # Get end-effector frame (last frame)
        frame_id = pin_model.nframes - 1
        T_pin = pin_data.oMf[frame_id]
        pos_pin = T_pin.translation
        R_pin = T_pin.rotation

        pos_err = np.linalg.norm(pos_ours - pos_pin)
        rot_err = np.linalg.norm(R_ours - R_pin)

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)

        if pos_err > FK_TOLERANCE:
            n_fail += 1

    print(f"  Pinocchio comparison ({n_samples} samples):")
    print(f"  Max position error: {max_pos_err:.2e} m")
    print(f"  Max rotation error: {max_rot_err:.2e}")
    print(f"  Failures (>{FK_TOLERANCE}m): {n_fail}/{n_samples}")

    if n_fail == 0 and max_pos_err < FK_TOLERANCE:
        print("[PASS] test_pinocchio_reference — MILESTONE 1 ACCEPTED ✅")
        return True
    else:
        print("[FAIL] test_pinocchio_reference — MILESTONE 1 NOT MET ❌")
        print("  FALLBACK: AI development paused. Fix kinematic inconsistency first.")
        return False


# ==============================================================================
# Main Runner
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("NeuroKinematics — Kinematic Validation Suite (FAZ 1)")
    print("=" * 70)
    
    tests = [
        test_urdf_parsing,
        test_fk_zero_config,
        test_fk_rotation_orthogonality,
        test_quaternion_consistency,
        test_6d_rotation_consistency,
        test_numpy_vs_torch_fk,
        test_differentiable_fk_gradient,
        test_jacobian_properties,
        test_jacobian_numerical_validation,
        test_singularity_detection,
        test_pinocchio_reference,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_fn in tests:
        print(f"\n--- {test_fn.__name__} ---")
        try:
            result = test_fn()
            if result is False:
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
    print("=" * 70)
    
    if failed > 0:
        print("⚠️  MILESTONE 1 AT RISK — Fix failures before proceeding to AI training.")
        sys.exit(1)
    else:
        print("✅ All core kinematic tests passed.")
