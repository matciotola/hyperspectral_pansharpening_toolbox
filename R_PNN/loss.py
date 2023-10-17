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

        self.cut_border = kernel.shape[0] // 2 // ratio

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

        loss_value = self.loss(outputs, labels[:, :, self.cut_border:-self.cut_border, self.cut_border:-self.cut_border])

        return loss_value


class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        Y = X * worst
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()
