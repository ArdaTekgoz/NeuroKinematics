"""
NeuroKinematics — PyTorch Dataset (Lazy HDF5 Loading)

Provides a PyTorch Dataset that reads from the HDF5 master file
with lazy loading for memory efficiency.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch as _torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False
    TorchDataset = object  # fallback base

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class IKDataset(TorchDataset):
    """
    PyTorch Dataset for IK training data stored in HDF5.

    Each sample returns:
        input: [target_position(3), target_rotation_6d(6), q_previous(n_joints)] = 15D
        target: [sin_q(n_joints), cos_q(n_joints)] = 12D

    Supports lazy loading from HDF5 for memory efficiency.

    Usage:
        dataset = IKDataset('data/kr6_dataset.h5', split='train')
        input_tensor, target_tensor = dataset[0]
    """

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        load_to_memory: bool = False,
    ):
        """
        Args:
            h5_path: Path to HDF5 dataset file.
            split: One of 'train', 'val', 'test'.
            load_to_memory: If True, load entire split into RAM.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.h5_path = h5_path
        self.split = split
        self._h5_file = None
        self._loaded = False

        # Read indices for the requested split
        with h5py.File(h5_path, 'r') as f:
            self.indices = f[f'splits/{split}_indices'][:]
            self.n_joints = int(f['metadata'].attrs['n_joints'])
            self.robot_name = str(f['metadata'].attrs['robot'])

        self._len = len(self.indices)

        if load_to_memory:
            self._load_to_memory()

    def _load_to_memory(self):
        """Load entire split data into RAM for faster access."""
        with h5py.File(self.h5_path, 'r') as f:
            idx = self.indices
            self._positions = f['inputs/target_position'][idx]
            self._rotations = f['inputs/target_rotation_6d'][idx]
            self._q_previous = f['inputs/q_previous'][idx]
            self._sin_q = f['outputs/sin_q'][idx]
            self._cos_q = f['outputs/cos_q'][idx]
            self._q = f['outputs/q'][idx]
            self._manipulability = f['physics/manipulability'][idx]
        self._loaded = True

    def _get_h5(self):
        """Lazy open HDF5 file (for lazy loading mode)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: [position(3), rotation_6d(6), q_prev(n_joints)]
            target_tensor: [sin_q(n_joints), cos_q(n_joints)]
        """
        if self._loaded:
            pos = self._positions[idx]
            rot = self._rotations[idx]
            q_prev = self._q_previous[idx]
            sin_q = self._sin_q[idx]
            cos_q = self._cos_q[idx]
        else:
            h5 = self._get_h5()
            real_idx = int(self.indices[idx])
            pos = h5['inputs/target_position'][real_idx]
            rot = h5['inputs/target_rotation_6d'][real_idx]
            q_prev = h5['inputs/q_previous'][real_idx]
            sin_q = h5['outputs/sin_q'][real_idx]
            cos_q = h5['outputs/cos_q'][real_idx]

        # Build input: [pos(3), rot6d(6), q_prev(6)] = 15D
        input_vec = np.concatenate([pos, rot, q_prev])
        # Build target: [sin_q(6), cos_q(6)] = 12D
        target_vec = np.concatenate([sin_q, cos_q])

        if HAS_TORCH:
            return _torch.tensor(input_vec, dtype=_torch.float32), \
                   _torch.tensor(target_vec, dtype=_torch.float32)
        else:
            return input_vec.astype(np.float32), target_vec.astype(np.float32)

    def get_raw_joints(self, idx: int) -> np.ndarray:
        """Get the raw joint angles q for a sample (for debugging)."""
        if self._loaded:
            return self._q[idx]
        h5 = self._get_h5()
        real_idx = int(self.indices[idx])
        return h5['outputs/q'][real_idx]

    def get_manipulability(self, idx: int) -> float:
        """Get manipulability value for a sample."""
        if self._loaded:
            return float(self._manipulability[idx])
        h5 = self._get_h5()
        real_idx = int(self.indices[idx])
        return float(h5['physics/manipulability'][real_idx])

    @property
    def input_dim(self) -> int:
        """Input dimension: 3 (pos) + 6 (rot6d) + n_joints (q_prev)."""
        return 3 + 6 + self.n_joints

    @property
    def output_dim(self) -> int:
        """Output dimension: 2 * n_joints (sin + cos)."""
        return 2 * self.n_joints

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass

    def __repr__(self):
        return (f"IKDataset(robot='{self.robot_name}', split='{self.split}', "
                f"n_samples={self._len}, input_dim={self.input_dim}, "
                f"output_dim={self.output_dim})")
