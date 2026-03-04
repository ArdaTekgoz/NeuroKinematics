import torch
import math

def dh_transform(a, alpha, d, theta):
    """
    Compute standard DH homogeneous transform.
    """
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    T = torch.zeros(theta.shape[0], 4, 4, device=theta.device)

    T[:,0,0] = ct
    T[:,0,1] = -st * ca
    T[:,0,2] = st * sa
    T[:,0,3] = a * ct

    T[:,1,0] = st
    T[:,1,1] = ct * ca
    T[:,1,2] = -ct * sa
    T[:,1,3] = a * st

    T[:,2,0] = 0
    T[:,2,1] = sa
    T[:,2,2] = ca
    T[:,2,3] = d

    T[:,3,0] = 0
    T[:,3,1] = 0
    T[:,3,2] = 0
    T[:,3,3] = 1

    return T


def forward_kinematics(q, dh_params):
    """
    q: (B,6) joint angles
    dh_params: list of 6 tuples (a, alpha, d, theta_offset)

    Returns:
        pose (B,7) -> x,y,z,qx,qy,qz,qw
    """

    B = q.shape[0]
    device = q.device

    T = torch.eye(4, device=device).unsqueeze(0).repeat(B,1,1)

    for i in range(6):
        a, alpha, d, theta_offset = dh_params[i]

        theta = q[:,i] + theta_offset

        Ti = dh_transform(
            torch.tensor(a, device=device).repeat(B),
            torch.tensor(alpha, device=device).repeat(B),
            torch.tensor(d, device=device).repeat(B),
            theta
        )

        T = torch.bmm(T, Ti)

    pos = T[:, :3, 3]
    R = T[:, :3, :3]

    quat = rotation_matrix_to_quaternion(R)

    return torch.cat([pos, quat], dim=1)


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    Returns (B,4) as qx,qy,qz,qw
    """
    B = R.shape[0]
    q = torch.zeros(B,4, device=R.device)

    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]

    for i in range(B):
        if trace[i] > 0:
            s = math.sqrt(trace[i] + 1.0) * 2
            q[i,3] = 0.25 * s
            q[i,0] = (R[i,2,1] - R[i,1,2]) / s
            q[i,1] = (R[i,0,2] - R[i,2,0]) / s
            q[i,2] = (R[i,1,0] - R[i,0,1]) / s
        else:
            # fallback
            q[i,3] = 1.0

    return q