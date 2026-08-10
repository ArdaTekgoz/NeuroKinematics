import torch
import numpy as np

def compute_position_error(pos_pred, pos_target):
    """
    Kartezyen uzayda pozisyon hatası (e_p) - Öklid mesafesi.
    pos_pred, pos_target: Tensor (B, 3) veya numpy array
    """
    if isinstance(pos_pred, torch.Tensor):
        return torch.norm(pos_pred - pos_target, dim=-1)
    return np.linalg.norm(pos_pred - pos_target, axis=-1)

def compute_orientation_error_6d(rot_pred, rot_target):
    """
    6D Continuous Rotation formatı için yönelim hatası.
    rot_pred, rot_target: Tensor (B, 6)
    """
    if isinstance(rot_pred, torch.Tensor):
        return torch.norm(rot_pred - rot_target, dim=-1)
    return np.linalg.norm(rot_pred - rot_target, axis=-1)

def compute_joint_limit_violation(q_pred, q_min, q_max):
    """
    Eklem limitlerinin dışına çıkma miktarını hesaplar.
    q_pred: Tensor (B, 6)
    q_min, q_max: Tensor (6,)
    """
    # Alt ve üst limitleri aşan kısımlar
    violation_lower = torch.relu(q_min - q_pred)
    violation_upper = torch.relu(q_pred - q_max)
    total_violation = torch.sum(violation_lower + violation_upper, dim=-1)
    return total_violation

def compute_jerk(q_seq, dt=1.0):
    """
    Yörünge sürekliliği için Jerk (Sarsıntı) hesaplar. 
    q_seq: Tensor (B, T, 6) (Zaman serisi)
    """
    # 3. türev
    if q_seq.shape[1] < 4:
        return torch.zeros(q_seq.shape[0])
    
    dq = (q_seq[:, 1:] - q_seq[:, :-1]) / dt
    ddq = (dq[:, 1:] - dq[:, :-1]) / dt
    dddq = (ddq[:, 1:] - ddq[:, :-1]) / dt
    
    # Jerk'in normu
    return torch.norm(dddq, dim=-1).mean(dim=1)

def recover_angles_from_sincos(sin_cos_tensor):
    """
    (B, 12) boyutundaki ağ çıktısını (sin, cos) gerçek açılara çevirir.
    Normalizasyon işlemi dahil edilmiştir.
    """
    B = sin_cos_tensor.shape[0]
    num_joints = sin_cos_tensor.shape[1] // 2
    
    s_val = sin_cos_tensor[:, 0::2]
    c_val = sin_cos_tensor[:, 1::2]
    
    # Normalizasyon
    norm = torch.sqrt(s_val**2 + c_val**2 + 1e-8)
    s_norm = s_val / norm
    c_norm = c_val / norm
    
    # Gerçek açı (atan2(y, x) -> atan2(sin, cos))
    angles = torch.atan2(s_norm, c_norm)
    return angles
