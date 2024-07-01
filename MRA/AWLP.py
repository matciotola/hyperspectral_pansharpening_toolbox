import torch

from Utils.spectral_tools import LPFilter
from Utils.imresize_bicubic import imresize


def AWLP(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    _, c, h, w = ms.shape

    mean_low = torch.mean(ms, dim=1, keepdim=True)

    img_intensity = ms / (mean_low + torch.finfo(ms.dtype).eps)

    pan = pan.repeat(1, c, 1, 1)

    pan_lp = imresize(imresize(pan, 1 / ratio), ratio)

    pan = (pan - torch.mean(pan, dim=(2, 3), keepdim=True)) * (
                torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(pan_lp, dim=(2, 3),
                                                                           keepdim=True)) + torch.mean(ms,
                                                                                                       dim=(2, 3),
                                                                                                       keepdim=True)

    pan_lpp = []
    for i in range(pan.shape[1]):
        pan_lpp.append(LPFilter(pan[:, i, None, :, :].type(ms.dtype), ratio))

    pan_lpp = torch.cat(pan_lpp, dim=1)

    details = pan - pan_lpp

    fused = details * img_intensity + ms

    return fused
