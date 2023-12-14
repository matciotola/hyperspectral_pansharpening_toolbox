import torch
from torch import nn
from math import floor, ceil
import numpy as np
from torch.nn import functional as func


def xcorr(img_1, img_2, half_width, device):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

    """
    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.double()
    img_2 = img_2.double()

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] - img_1_cum[:, :, 2 * w:, :-2 * w] +
                img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] - img_2_cum[:, :, 2 * w:, :-2 * w] +
                img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1 ** 2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2 ** 2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] - ij_cum[:, :, 2 * w:, :-2 * w] +
                   ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] - i2_cum[:, :, 2 * w:, :-2 * w] +
                   i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] - j2_cum[:, :, 2 * w:, :-2 * w] +
                   j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ij_tot = torch.clip(sig2_ij_tot, ep, sig2_ij_tot.max().item())
    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    corr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return corr


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
        X_corr = torch.clamp(xcorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        Y = X * worst
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()
