import torch
import torch.nn.functional as F
from torch import nn


from . import metrics_rr as mt
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
from .cross_correlation import xcorr_torch
from Utils.imresize_bicubic import imresize as resize

def downgrade(img, kernel, ratio):
    img = F.conv2d(img, kernel.type(img.dtype).to(img.device), padding='same', groups=img.shape[1])
    ratio = int(ratio)
    img = img[:, :, 3::ratio, 3::ratio]
    return img


class ReproErgas(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproErgas, self).__init__()
        self.ratio = ratio
        self.ERGAS = mt.ERGAS(self.ratio)
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'

        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.ERGAS(downgraded_outputs, labels)


class ReproSAM(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproSAM, self).__init__()
        self.ratio = ratio
        self.SAM = mt.SAM()
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.SAM(downgraded_outputs, labels)


class ReproQ2n(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproQ2n, self).__init__()
        self.ratio = ratio
        self.Q2n = mt.Q2n()
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.Q2n(downgraded_outputs, labels)


class ReproQ(nn.Module):
    def __init__(self, ratio, lana=False, device='cuda'):
        super(ReproQ, self).__init__()
        self.ratio = ratio

        if ratio == 2:
            sensor = 'S2-20'
            nbands = 6
        elif ratio == 6:
            sensor = 'S2-60'
            nbands = 3
        else:
            sensor = 'S2-60'
            nbands = 3
        if lana and ratio == 6:
            sensor = 'S2-60_bis'

        self.Q = mt.Q(nbands, device)
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.Q(downgraded_outputs, labels)


class LSR(nn.Module):
    def __init__(self):
        # Class initialization
        super(LSR, self).__init__()

    @staticmethod
    def forward(outputs, pan):
        pan = pan.double()
        outputs = outputs.double()

        pan_flatten = torch.flatten(pan, start_dim=-2).transpose(2, 1)
        fused_flatten = torch.flatten(outputs, start_dim=-2).transpose(2, 1)
        with torch.no_grad():
            alpha = (fused_flatten.pinverse() @ pan_flatten)[:, :, :, None]
        i_r = torch.sum(outputs * alpha, dim=1, keepdim=True)

        err_reg = pan - i_r

        cd = 1 - (torch.var(err_reg, dim=(1, 2, 3)) / torch.var(pan, dim=(1, 2, 3)))

        return cd


class D_s(nn.Module):
    def __init__(self, nbands, ratio=4, q=1, q_block_size=32):
        super(D_s, self).__init__()
        self.Q_high = mt.Q(nbands, q_block_size)
        self.Q_low = mt.Q(nbands, q_block_size // ratio)
        self.nbands = nbands
        self.ratio = ratio
        self.q = q

    def forward(self, outputs, pan, ms):
        pan = pan.repeat(1, self.nbands, 1, 1)
        pan_lr = resize(pan, scale=1 / self.ratio)

        q_high = self.Q_high(outputs, pan)
        q_low = self.Q_low(ms, pan_lr)

        ds = torch.sum(abs(q_high - q_low) ** self.q, dim=1)

        ds = (ds / self.nbands) ** (1 / self.q)

        return ds

class D_sR(nn.Module):
    def __init__(self):
        super(D_sR, self).__init__()
        self.metric = LSR()

    def forward(self, outputs, pan):
        lsr = torch.mean(self.metric(outputs, pan))
        return 1.0 - lsr

class D_rho(nn.Module):

    def __init__(self, sigma):
        # Class initialization
        super(D_rho, self).__init__()

        # Parameters definition:
        self.scale = sigma // 2

    def forward(self, outputs, labels):
        Y = torch.ones((outputs.shape[0], outputs.shape[1], outputs.shape[2], outputs.shape[3], labels.shape[1]),
                       device=outputs.device)

        for i in range(labels.shape[1]):
            Y[:, :, :, :, i] = torch.clamp(xcorr_torch(outputs,
                                                       torch.unsqueeze(labels[:, i, :, :],
                                                                       1),
                                                       self.scale, outputs.device), min=-1.0)

        Y = torch.amax(Y, -1)
        Y = torch.clip(Y, -1, 1)
        X = 1.0 - Y
        Lxcorr = torch.mean(X)

        return Lxcorr, torch.mean(X, dim=(2, 3))
