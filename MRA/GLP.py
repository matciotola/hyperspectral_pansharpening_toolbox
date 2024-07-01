import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter

from Utils.spectral_tools import mtf, LPfilterGauss
from Utils.pansharpening_aux_tools import batch_cov, estimation_alpha
from Utils.interpolator_tools import ideal_interpolator


def MTF_GLP_FS(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    sensor = ordered_dict.sensor
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
    sensor = ordered_dict.sensor
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2, 3), keepdim=True)) * (
            torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(LPfilterGauss(bands_hr, ratio), dim=(2, 3),
                                                                       keepdim=True)) + torch.mean(ms,
                                                                                                   dim=(2, 3),
                                                                                                   keepdim=True)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    # bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio)
    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    fused = ms * torch.clip((bands_hr / (bands_hr_lr + torch.finfo(ms.dtype).eps)), 0, 10.0)
    return fused


def MTF_GLP_HPM_R(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    sensor = ordered_dict.sensor
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


if __name__ == '__main__':
    from scipy import io
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    import numpy as np
    from recordclass import recordclass
    from Utils.interpolator_tools import interp23tap
    # temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_1_zoom.mat')
    temp = io.loadmat('/home/matteo/Desktop/RR1_Barcelona.mat')

    ms_lr = temp['I_MS_LR'].astype(np.float64)
    ms = temp['I_MS'].astype(np.float64)
    pan = temp['I_PAN'].astype(np.float64)
    ratio = 6

    ms_lr = torch.tensor(np.moveaxis(ms_lr, -1, 0)[None, :, :, :])
    ms = torch.tensor(np.moveaxis(ms, -1, 0)[None, :, :, :])
    pan = torch.tensor(pan[None, None, :, :])

    ord_dic = {'ms': ms, 'pan': pan, 'ms_lr': ms_lr, 'ratio': ratio, 'dataset': 'PRISMA'}

    exp_input = recordclass('exp_info', ord_dic.keys())(*ord_dic.values())

    fused = MTF_GLP_HPM(exp_input)
    plt.figure()
    plt.imshow(fused[0, 0, :, :].detach().numpy(), cmap='gray')
    plt.show()