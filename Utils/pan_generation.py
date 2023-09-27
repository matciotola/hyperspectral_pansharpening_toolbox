import torch
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
from torch.nn.functional import conv2d, pad

def selection(pan, ms, ratio, sat_name='S2-10'):

    h = mtf_kernel_to_torch(gen_mtf(ratio, sat_name)).to(pan.device)

    # Decimation
    pan_lpf = conv2d(pad(pan, (h.shape[-1] // 2, h.shape[-1] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode='replicate'), h, groups=pan.shape[1])
    pan_lr = pan_lpf[:, :, 1::ratio, 1::ratio]

    # CorrCoeff Calculation

    corr_coeff_global = []

    for i in range(pan_lr.shape[1]):
        selected_high = pan_lr[:, i, :, :].flatten()[None, :]
        corr_coeff_selected_high = []
        for j in range(ms.shape[1]):
            selected_low = ms[:, j, :, :].flatten()[None, :]
            X = torch.cat([selected_high, selected_low], dim=0)
            corr_coeff_selected_high.append(torch.corrcoef(X)[0,1])
        corr_coeff_global.append(torch.Tensor(corr_coeff_selected_high))

    corr_coeff_global = torch.vstack(corr_coeff_global)

    _, selected_bands = torch.max(corr_coeff_global, dim=0)
    high_composed = pan[:, selected_bands, :, :]

    return high_composed, selected_bands

def vectorized_alpha_estimation (a, b):

    a_fl = torch.flatten(a, start_dim=-2).transpose(2, 1)
    b_fl = torch.flatten(b, start_dim=-2).transpose(2, 1)
    alpha = torch.linalg.lstsq(a_fl, b_fl)
    alpha = alpha.solution[:, :, :, None]

    return alpha
def synthesize(pan, ms,  ratio, sat_name='S2-10'):

    scale = 1 / ratio

    h = mtf_kernel_to_torch(gen_mtf(ratio, sat_name)).type(pan.dtype).to(pan.device)
    pan_lpf = conv2d(
                            pad(pan, (h.shape[-1] // 2,
                                             h.shape[-1] // 2,
                                             h.shape[-1] // 2,
                                             h.shape[-1] // 2),
                                mode='replicate'),
                            h,
                            groups=pan.shape[1])
    pan_lr = pan_lpf[:, :, 1::ratio, 1::ratio]

    padding_high_lr = torch.ones((pan_lr.shape[0], 1, pan_lr.shape[2], pan_lr.shape[3]), dtype=pan_lr.dtype,
                              device=pan_lr.device)
    pan_lr_plus = torch.cat([padding_high_lr, pan_lr], dim=1)

    alpha = vectorized_alpha_estimation(pan_lr_plus.double(), ms.double()).float()

    pan_exp = torch.ones((pan.shape[0], pan.shape[1] + 1, pan.shape[2], pan.shape[3]),
                                 dtype=pan.dtype,
                                 device=pan.device)

    pan_exp[:, 1:, :, :] = pan

    synthesized_bands = []
    for i in range(alpha.shape[2]):
        a = alpha[:, :, i, :, None]
        synthesized_bands.append(torch.sum(pan_exp * a, dim=1, keepdim=True))

    synthesized_bands = torch.cat(synthesized_bands, dim=1)

    selected_bands = list(range(ms.shape[1]))

    return synthesized_bands, selected_bands

