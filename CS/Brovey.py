import torch

from Utils.spectral_tools import LPfilterGauss
from Utils.pansharpening_aux_tools import estimation_alpha


def BT_H(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    min_ms = torch.amin(ms, dim=(2, 3), keepdim=True)

    pan_lp = LPfilterGauss(pan, ratio)
    alphas = estimation_alpha(ms, pan_lp)

    img = torch.sum((ms - min_ms) * alphas, dim=1, keepdim=True)

    img_hr = (pan - torch.mean(pan_lp, dim=(2, 3), keepdim=True)) * (
                torch.std(img, dim=(2, 3), keepdim=True) / torch.std(pan_lp, dim=(2, 3),
                                                                     keepdim=True)) + torch.mean(img, dim=(2, 3),
                                                                                                 keepdim=True)

    ms_minus_min = ms - min_ms
    ms_minus_min = torch.clip(ms_minus_min, 0, ms_minus_min.max())

    fused = ms_minus_min * (img_hr / (img + torch.finfo(torch.float64).eps)).repeat(1, ms.shape[1], 1,
                                                                                           1) + min_ms

    return fused
