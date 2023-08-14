import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import os
import gc
import librosa
import PIL.Image as Image
import numpy as np
from pathlib import Path
import torch
import math
import tgm


### rotational conversion
def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert an angle axis to a quaternion.

    DECA project adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}"
                         .format(angle_axis.shape))
    # unpack input and compute conversion
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    '''  same as batch_matrix2axis
    Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
             x: pitch. positive for looking down.
            y: yaw. positive for looking left.
            z: roll. positive for tilting head right.
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
def batch_axis2euler(r):
    return rot_mat_to_euler(batch_rodrigues(r))




def angle_axis_to_quaternion_numpy(angle_axis):
    """Convert an angle axis to a quaternion.

    DECA project adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (np.ndarray): numpy array with angle axis.

    Return:
        np.ndarray: numpy array with quaternion.

    Shape:
        - Input: `(N, 3)`
        - Output: `(N, 4)`

    Example:
        >>> angle_axis = np.random.rand(2, 3)
        >>> quaternion = angle_axis_to_quaternion(angle_axis)
    """
    if not isinstance(angle_axis, np.ndarray):
        raise TypeError("Input type is not a np.ndarray. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a numpy array of shape Nx3 or 3. Got {}"
                         .format(angle_axis.shape))
    # unpack input and compute conversion
    a0 = angle_axis[..., 0:1]
    a1 = angle_axis[..., 1:2]
    a2 = angle_axis[..., 2:3]
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2

    theta = np.sqrt(theta_squared)
    half_theta = theta * 0.5

    mask = theta_squared > 0.0
    ones = np.ones_like(half_theta)

    k_neg = 0.5 * ones
    k_pos = np.sin(half_theta) / theta
    k = np.where(mask, k_pos, k_neg)
    w = np.where(mask, np.cos(half_theta), ones)

    quaternion = np.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return np.concatenate([w, quaternion], axis=-1)



def quaternion_to_euler_xyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to Euler angles in XYZ order.

    Args:
        quaternion (torch.Tensor): tensor with quaternion.

    Return:
        torch.Tensor: tensor with Euler angles in XYZ order.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        #>>> quaternion = torch.rand(2, 4)  # Nx4
        #>>> euler_xyz = quaternion_to_euler_xyz(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))

    w, x, y, z = quaternion[..., 0:1], quaternion[..., 1:2], quaternion[..., 2:3], quaternion[..., 3:4]

    # Calculate Euler angles from quaternion
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.tensor(math.pi / 2, device=sinp.device), torch.asin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.cat([roll, pitch, yaw], dim=-1)

#
# data_con = np.load("data/incoming_data/machine_learning_output_sd.npy")
# print(data_con[0,0,0,50:].shape)
#
#
# posecode_tensor = torch.from_numpy(data_con[0,0,0,50:])
#
# glob_qat = angle_axis_to_quaternion(posecode_tensor[:3]) # Nx4
# jaw_qat = angle_axis_to_quaternion(posecode_tensor[3:]) # Nx4
#
#
#
# print("glob_qat",glob_qat, "jaw_qat", jaw_qat)
#
# glob_euler_xyz = quaternion_to_euler_xyz(glob_qat)
# jaw_euler_xyz = quaternion_to_euler_xyz(jaw_qat)
#
# print("glob_euler_xyz",glob_euler_xyz, "jaw_euler_xyz", jaw_euler_xyz)