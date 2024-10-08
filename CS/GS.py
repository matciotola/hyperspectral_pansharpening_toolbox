import torch

from Utils.spectral_tools import LPFilterPlusDecTorch
from Utils.pansharpening_aux_tools import estimation_alpha


def GSA(ordered_dict):

    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ms_lr = torch.clone(ordered_dict.ms_lr)
    ratio = ordered_dict.ratio

    ms_interp_minus_avg = ms - torch.mean(ms, dim=(2, 3), keepdim=True)
    ms_minus_avg = ms_lr - torch.mean(ms_lr, dim=(2, 3), keepdim=True)
    pan_minus_avg = pan - torch.mean(pan, dim=(2, 3), keepdim=True)

    pan_minus_avg = LPFilterPlusDecTorch(pan_minus_avg, ratio)
    B, _, h, w = ms_minus_avg.shape
    _, _, H, W = ms_interp_minus_avg.shape
    alpha_input = torch.cat([ms_minus_avg,
                             torch.ones((B, 1, h, w), dtype=ms_minus_avg.dtype,
                                        device=ms_minus_avg.device)], dim=1)
    alphas = estimation_alpha(alpha_input, pan_minus_avg)
    alpha_out = torch.cat([ms_interp_minus_avg,
                           torch.ones((B, 1, H, W), dtype=ms_minus_avg.dtype,
                                      device=ms_minus_avg.device)], dim=1)
    img = torch.sum(alpha_out * alphas, dim=1, keepdim=True)

    ms_minus_int = img - torch.mean(img, dim=(2, 3))

    g = torch.ones(1, ms.shape[1] + 1, 1)
    for i in range(ms.shape[1]):
        h = ms_interp_minus_avg[:, i:i + 1, :, :]
        cc = torch.cov(torch.flatten(torch.cat([ms_minus_int[0], h[0]], dim=0), start_dim=-2))
        g[:, i + 1, :] = cc[0, 1] / ms_minus_int.var()

    pan = pan - torch.mean(pan, dim=(2, 3))

    delta = pan - ms_minus_int
    delta_flatten = torch.flatten(delta.transpose(2, 3), start_dim=-2)
    delta_r = delta_flatten.repeat(1, ms.shape[1] + 1, 1)

    v1 = torch.flatten(ms_minus_int.transpose(-2, -1), start_dim=-2)
    v2 = torch.flatten(ms_interp_minus_avg.transpose(-2, -1), start_dim=-2)

    V = torch.cat([v1, v2], dim=1)

    gm = g[0, None:, 0, None].repeat(1, 1, V.shape[-1])

    V_hat = V + delta_r * gm

    V_hat = torch.reshape(V_hat[:, 1:, :], (ms.shape[0], ms.shape[1], ms.shape[3], ms.shape[2])).transpose(2, 3)

    fused = V_hat - V_hat.mean(dim=(-2, -1), keepdim=True) + ms.mean(dim=(-2, -1), keepdim=True)

    return fused
