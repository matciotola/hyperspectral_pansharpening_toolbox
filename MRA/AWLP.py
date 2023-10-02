import torch

from Utils.spectral_tools import LPFilter
from Utils.imresize_bicubic import imresize


def AWLP(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    bs, c, h, w = ms.shape

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


if __name__ == '__main__':
    from scipy import io
    import matplotlib
    matplotlib.use('QT5Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    from recordclass import recordclass
    from Utils.interpolator_tools import interp23tap
    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_1_zoom.mat')

    ms_lr = temp['I_MS_LR'].astype(np.float64)
    ms = interp23tap(ms_lr, 4)
    pan = temp['I_PAN'].astype(np.float64)
    ratio = 4

    ms_lr = torch.tensor(np.moveaxis(ms_lr, -1, 0)[None, :, :, :])
    ms = torch.tensor(np.moveaxis(ms, -1, 0)[None, :, :, :])
    pan = torch.tensor(pan[None, None, :, :])

    ord_dic = {'ms': ms, 'pan': pan, 'ms_lr': ms_lr, 'ratio': ratio}

    exp_input = recordclass('exp_info', ord_dic.keys())(*ord_dic.values())

    fused = AWLP(exp_input)
    fused = torch.clip(fused, 0, 2048.0)
    plt.figure()
    plt.imshow(fused[0, 0, :, :].detach().cpu().numpy(), cmap='gray', clim=[0, fused.max()])
    plt.show()