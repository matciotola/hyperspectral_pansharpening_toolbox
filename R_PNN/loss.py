import torch
from torch import nn
from math import floor, ceil
from Metrics.cross_correlation import xcorr_torch as ccorr
import numpy as np


class SpectralLoss(nn.Module):
    def __init__(self, mtf, ratio, device):

        # Class initialization
        super(SpectralLoss, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):

        outputs = self.depthconv(outputs)
        outputs = outputs[:, :, 3::self.ratio, 3::self.ratio]

        loss_value = self.loss(outputs, labels)

        return loss_value


class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        x_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        d_rho = 1.0 - x_corr

        with torch.no_grad():
            d_rho = torch.mean(d_rho)

        worst = d_rho.gt(xcorr_thr)
        d_rho_with_threshold = d_rho * worst

        d_rho_with_threshold = torch.mean(d_rho_with_threshold)

        return d_rho_with_threshold, d_rho.item()
