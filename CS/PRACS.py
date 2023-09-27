import torch

from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter

from Utils.pansharpening_aux_tools import regress


def PRACS(ordered_dict, beta=0.95):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    B, C, H, W = ms.shape

    ms_hm = (ms - torch.mean(ms, dim=(2, 3), keepdim=True) + torch.mean(pan, dim=(2, 3),
                                                                                             keepdim=True) / torch.std(
        pan, dim=(2, 3), keepdim=True) * torch.std(ms, dim=(2, 3), keepdim=True)) * torch.std(pan,
                                                                                                            dim=(2, 3),
                                                                                                            keepdim=True) / torch.std(
        ms, dim=(2, 3), keepdim=True)
    ms_hm = torch.clip(ms_hm, 0, ms_hm.max())

    pan_lp = resize(
        resize(pan, [pan.shape[2] // ratio, pan.shape[3] // ratio], interpolation=Inter.BICUBIC,
               antialias=True), [pan.shape[2], pan.shape[3]], interpolation=Inter.BICUBIC, antialias=True)

    bb = torch.cat([torch.ones((B, 1, H, W), dtype=ms_hm.dtype, device=ms_hm.device), ms_hm],
                   dim=1)
    bb = torch.flatten(bb, start_dim=2).transpose(1, 2)
    pan_lp_f = torch.flatten(pan_lp, start_dim=2).transpose(1, 2)
    alpha = regress(pan_lp_f, bb)

    aux = torch.matmul(bb, alpha)

    img = torch.reshape(aux, (B, 1, H, W))

    corr_coeffs = []
    for b in range(B):
        corr_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img[b, :, :, :], ms_hm[b, c, None, :, :]], dim=0), start_dim=1)
            corr_bands.append(torch.corrcoef(stack)[0, 1])
        corr_bands = torch.vstack(corr_bands)
        corr_coeffs.append(corr_bands[None, :, :])

    corr_coeffs = torch.vstack(corr_coeffs)[:, :, :, None]

    img_h = corr_coeffs * pan.repeat(1, C, 1, 1) + (1 - corr_coeffs) * ms_hm

    img_h_lp = resize(
        resize(img_h, [pan.shape[2] // ratio, pan.shape[3] // ratio], interpolation=Inter.BICUBIC,
               antialias=True), [pan.shape[2], pan.shape[3]], interpolation=Inter.BICUBIC, antialias=True)
    img_h_lp_f = torch.flatten(img_h_lp, start_dim=2)
    gamma = []
    for i in range(C):
        aux = img_h_lp_f[:, i, None, :].transpose(1, 2)
        gamma.append(regress(aux, bb))
    gamma = torch.cat(gamma, dim=-1)

    img_prime = []
    for i in range(C):
        aux = torch.bmm(bb, gamma[:, :, i, None])
        img_prime.append(torch.reshape(aux, (B, 1, H, W)))

    img_prime = torch.cat(img_prime, 1)

    delta = img_h - img_prime - (
                torch.mean(img_h, dim=(2, 3), keepdim=True) - torch.mean(img_prime, dim=(2, 3), keepdim=True))

    aux3 = torch.mean(torch.std(ms, dim=(2, 3)), dim=1)

    w = []
    for b in range(B):
        w_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img_prime[b, c, None, :, :], ms[b, c, None, :, :]], dim=0),
                                  start_dim=1)
            w_bands.append(torch.corrcoef(stack)[0, 1])
        w_bands = torch.vstack(w_bands)
        w.append(w_bands[None, :, :, None])
    w = torch.vstack(w)
    w = beta * w * torch.std(ms, dim=(2, 3), keepdim=True) / aux3

    L_i = []
    for b in range(B):
        L_i_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img[b, 0, None, :, :], ms[b, c, None, :, :]], dim=0), start_dim=1)

            rho = torch.corrcoef(stack)[0, 1]
            aux = 1 - abs(1 - rho * ms[b, c, None, :, :] / img_prime[b, c, None, :, :])

            L_i_bands.append(torch.reshape(aux, (1, 1, H, W)))
        L_i_bands = torch.cat(L_i_bands, 1)
        L_i.append(L_i_bands)
    L_i = torch.cat(L_i, 0)

    det = w * L_i * delta
    fused = ms + det

    return fused

