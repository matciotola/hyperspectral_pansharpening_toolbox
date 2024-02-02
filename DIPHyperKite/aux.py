import torch
from torch.utils.data import Dataset
from Utils.load_save_tools import open_mat
import numpy as np
from torch import nn
import math
class TrainingDatasetKite(Dataset):
    def __init__(self, img_paths, priors, norm):
        super(TrainingDatasetKite, self).__init__()

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


class TestDatasetKite(Dataset):
    def __init__(self, pan, ms_lr):
        super(TestDatasetKite, self).__init__()

        self.pan = pan
        self.ms_lr = ms_lr

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.ms_lr[index], 0, 0


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, spatial_size, noise_type='u', var=1./10):

    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input

def initialize_weights(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()