from memory_profiler import profile
from tqdm import tqdm

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import os
import gc
import librosa
import PIL.Image as Image
import numpy as np
from pathlib import Path
import torch

def load_mfcc(audio_path, num_frames):
    waveform, sample_rate = librosa.load('{}'.format(audio_path), sr=16000)
    win_len = int(0.025*sample_rate)
    hop_len = int(0.010*sample_rate)
    fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
    S_dB = librosa.feature.mfcc(y=waveform, sr=sample_rate,n_mfcc=128, hop_length=hop_len)

    ## do some resizing to match frame rate
    im = Image.fromarray(S_dB)
    _, feature_dim = im.size
    scale_four = num_frames*4
    im = im.resize((scale_four, feature_dim), Image.ANTIALIAS)
    S_dB = np.array(im)
    return S_dB

def stream_mfcc(audio_path, num_frames):
    waveform, sample_rate = librosa.load('{}'.format(audio_path), sr=16000)
    win_len = int(0.025*sample_rate)
    hop_len = int(0.010*sample_rate)
    fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
    S_dB = librosa.feature.mfcc(y=waveform, sr=sample_rate,n_mfcc=128, hop_length=hop_len)

    ## do some resizing to match frame rate
    im = Image.fromarray(S_dB)
    _, feature_dim = im.size
    scale_four = num_frames*4
    im = im.resize((scale_four, feature_dim), Image.ANTIALIAS)
    S_dB = np.array(im)
    return S_dB


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
        #>>> angle_axis = torch.rand(2, 4)  # Nx4
        #>>> quaternion = tgm.angle_axis_to_quaternion(angle_axis)  # Nx3
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



def dir_or_file(args, path):
    if os.path.isdir(args.video_pth):
        print("\nIt is a directory")
        return 1
    elif os.path.isfile(args.video_pth):
        print("\nIt is a normal file")
        return 2
    else:
        raise argparse.ArgumentTypeError(f"readable_path:{path} is not a valid dir or a path")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

