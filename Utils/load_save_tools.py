import torch
from scipy import io
import numpy as np
from Utils.imresize_bicubic import imresize
# from Utils.interpolator_tools import interp23tap

def open_mat(path):
    # Open .mat file
    dic_file = io.loadmat(path)

    # Extract fields and convert them in float32 numpy arrays
    pan_np = dic_file['I_PAN'].astype(np.float64)
    ms_lr_np = dic_file['I_MS_LR'].astype(np.float64)
    ms_np = dic_file['I_MS'].astype(np.float64)

    if 'I_GT' in dic_file.keys():
        gt_np = dic_file['I_GT'].astype(np.float64)
        gt = torch.from_numpy(np.moveaxis(gt_np, -1, 0)[None, :, :, :])
    else:
        gt = None

    # Convert numpy arrays to torch tensors
    ms_lr = torch.from_numpy(np.moveaxis(ms_lr_np, -1, 0)[None, :, :, :])
    pan = torch.from_numpy(pan_np[None, None, :, :])
    ms = torch.from_numpy(np.moveaxis(ms_np, -1, 0)[None, :, :, :])
    wavelenghts = torch.squeeze(torch.from_numpy(dic_file['wavelengths']).float())
    overlap = torch.arange(0, dic_file['overlap'].item())

    return pan, ms_lr, ms, gt, wavelenghts, overlap


def save_mat(image, path):
    io.savemat(path, {'I_MS': image})
    return
