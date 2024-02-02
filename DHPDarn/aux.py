import numpy as np
import torch
from torch.utils.data import Dataset
from Utils.load_save_tools import open_mat

class TrainingDatasetDarn(Dataset):
    def __init__(self, img_paths, priors, norm):
        super(TrainingDatasetDarn, self).__init__()

        pan = []
        gt = []

        for i in range(len(img_paths)):
            pan_single, _, _, gt_single, _, _ = open_mat(img_paths[i])
            pan.append(pan_single.float())
            gt.append(gt_single.float())

        pan = torch.cat(pan, 0)
        gt = torch.cat(gt, 0)

        pan = norm(pan)
        gt = norm(gt)

        self.pan = pan
        self.gt = gt
        self.priors = priors

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.priors[index], self.gt[index]


class TestDatasetDHP(Dataset):
    def __init__(self, pan, ms_lr):
        super(TestDatasetDHP, self).__init__()

        self.pan = pan
        self.ms_lr = ms_lr

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.ms_lr[index], 0, 0

def get_lanczos_kernel(factor, phase, kernel_width, support):

    # factor  = float(factor)
    if phase == 0.5:
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    center = (kernel_width + 1) / 2.

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):

            if phase == 0.5:
                di = abs(i + 0.5 - center) / factor
                dj = abs(j + 0.5 - center) / factor
            else:
                di = abs(i - center) / factor
                dj = abs(j - center) / factor

            pi_sq = np.pi * np.pi

            val = 1
            if di != 0:
                val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                val = val / (np.pi * np.pi * di * di)

            if dj != 0:
                val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                val = val / (np.pi * np.pi * dj * dj)

            kernel[i - 1][j - 1] = val

    kernel /= kernel.sum()

    return kernel
