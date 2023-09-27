import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter
import matplotlib

matplotlib.use('Qt5Agg')

from Utils.spectral_tools import mtf, LPfilterGauss
from Utils.pansharpening_aux_tools import batch_cov, estimation_alpha
from Utils.interpolator_tools import ideal_interpolator


def MTF_GLP(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    sensor = ordered_dict.dataset
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)
    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2, 3), keepdim=True)) * (
            torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(LPfilterGauss(bands_hr, ratio), dim=(2, 3),
                                                                       keepdim=True)) + torch.mean(ms,
                                                                                                   dim=(2, 3),
                                                                                                   keepdim=True)

    pan_lp = mtf(bands_hr.float(), sensor, ratio)
    pan_lr_lr = resize(pan_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT,
                              antialias=False)
    # pan_lr = interp23tap_torch(pan_lr_lr, ratio)
    pan_lr = ideal_interpolator(pan_lr_lr, ratio)

    fused = ms + bands_hr - pan_lr

    return fused


def MTF_GLP_FS(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    # sensor = ordered_dict.sensor
    sensor = ordered_dict.dataset
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)

    # bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio)
    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    low_covs = []
    high_covs = []

    for i in range(ms.shape[1]):
        points_lr = torch.cat([ms[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_lr = torch.flatten(points_lr, start_dim=2)

        points_hr = torch.cat([bands_hr_lr[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_hr = torch.flatten(points_hr, start_dim=2)

        low_covs.append(batch_cov(points_lr.transpose(1, 2))[:, None, :, :])
        high_covs.append(batch_cov(points_hr.transpose(1, 2))[:, None, :, :])

    low_covs = torch.cat(low_covs, dim=1)
    high_covs = torch.cat(high_covs, dim=1)

    gamma = low_covs[:, :, 0, 1] / high_covs[:, :, 0, 1]
    gamma = gamma[:, :, None, None]
    fused = ms + gamma * (bands_hr - bands_hr_lr)

    return fused


def MTF_GLP_HPM(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    # sensor = ordered_dict.sensor
    sensor = ordered_dict.dataset
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2, 3), keepdim=True)) * (
            torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(LPfilterGauss(bands_hr, ratio), dim=(2, 3),
                                                                       keepdim=True)) + torch.mean(ms,
                                                                                                   dim=(2, 3),
                                                                                                   keepdim=True)

    bands_hr_lp = mtf(bands_hr.float(), sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    # bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio)
    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    fused = ms * (bands_hr / (bands_hr_lr + torch.finfo(ms.dtype).eps))

    return fused


def MTF_GLP_HPM_H(ordered_dict, decimation=True):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    # sensor = ordered_dict.sensor
    sensor = ordered_dict.dataset
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    min_ms = torch.amin(ms, dim=(2, 3), keepdim=True)
    pan_lp = LPfilterGauss(pan, ratio)

    inp = torch.cat([torch.ones(pan_lp.shape, dtype=pan_lp.dtype, device=pan_lp.device),
                     ms],
                    dim=1)

    alpha = estimation_alpha(inp, pan_lp)

    alpha_p = torch.bmm(torch.squeeze(alpha, -1).transpose(1, 2), torch.squeeze(
        torch.cat([torch.ones((bs, 1, 1, 1), device=alpha.device, dtype=alpha.dtype), min_ms], dim=1),
        -1).float())[:, :, :, None]

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)

    if decimation:
        bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
        # bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio)
        bands_hr_lp = ideal_interpolator(bands_hr_lr_lr, ratio)

    bands_hr_pl = (bands_hr - alpha_p) / (bands_hr_lp - alpha_p + torch.finfo(bands_hr_lp.dtype).eps)

    ms_l = ms - min_ms

    fused = ms_l * bands_hr_pl + min_ms

    return fused


def MTF_GLP_HPM_R(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    # sensor = ordered_dict.sensor
    sensor = ordered_dict.dataset
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    # bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio)
    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    g = []
    for i in range(c):
        inp = torch.flatten(torch.cat([ms[:, i, None, :, :], bands_hr_lr[:, i, None, :, :]], dim=1),
                            start_dim=2).transpose(1, 2)
        C = batch_cov(inp)
        g.append((C[:, 0, 1] / C[:, 1, 1])[:, None])

    g = torch.cat(g, dim=1)[:, :, None, None]
    cb = torch.mean(ms, dim=(2, 3), keepdim=True) / g - torch.mean(bands_hr, dim=(2, 3), keepdim=True)

    fused = ms * (bands_hr + cb) / (bands_hr_lr + cb + torch.finfo(ms.dtype).eps)

    return fused
