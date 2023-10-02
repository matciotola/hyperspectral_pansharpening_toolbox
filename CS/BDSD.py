import torch
from torch.nn.functional import pad
from torchvision.transforms import InterpolationMode as Inter
from torchvision.transforms.functional import resize

from Utils.imresize_bicubic import imresize

from Utils.spectral_tools import mtf, mtf_pan


def BDSD(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    ms_lr_lp, ms_lr, pan_lr = prepro_BDSD(ms, pan, ratio, ms.shape[-1], ordered_dict.dataset)

    gamma = gamma_calculation_BDSD(ms_lr_lp, ms_lr, pan_lr, ratio, ms.shape[-1])
    fused = fuse_BDSD(ms, pan, gamma, ratio, ms.shape[-1])

    return fused


def prepro_BDSD(ms, pan, ratio, block_size, sensor='PRISMA'):
    assert (block_size % 2 == 0), f"block size for local estimation must be even"
    assert (block_size > 1), f"block size for local estimation must be positive"
    assert (block_size % ratio == 0), f"block size must be multiple of ratio"

    _, _, N, M = pan.shape
    _, _, n, m = ms.shape
    assert (N % block_size == 0) and (
            M % block_size == 0), f"height and widht of 10m bands must be multiple of the block size"

    #ms = ms.float()
    #pan = pan.float()
    starting = ratio // 2

    pan_lp = mtf_pan(pan, sensor, ratio)
    pan_lr = pan_lp[:, :, starting::ratio, starting::ratio]
    # ms_lr = resize(ms, [n // ratio, m // ratio], interpolation=Inter.BICUBIC, antialias=True)
    ms_lr = imresize(ms, scale=1 / ratio)
    ms_lr_lp = mtf(ms_lr, sensor, ratio)

    return ms_lr_lp, ms_lr, pan_lr


def gamma_calculation_BDSD(ms_lr_lp, ms_lr, pan_lr, ratio, block_size):
    alg_input = torch.cat([ms_lr_lp, ms_lr, pan_lr], dim=1)
    gamma = blockproc(alg_input, (int(block_size // ratio), int(block_size // ratio)), estimate_gamma_cube, block_size,
                      ratio)
    return gamma


def fuse_BDSD(ms, pan, gamma, ratio, block_size):
    ## Fusion

    inputs = torch.cat([ms, pan, gamma], dim=1)

    fused = blockproc(inputs, (block_size, block_size), compH_injection, block_size, ratio)

    return fused


def blockproc(A, dims, func, S, ratio):
    results_row = []
    for y in range(0, A.shape[-2], dims[0]):
        results_cols = []
        for x in range(0, A.shape[-1], dims[1]):
            results_cols.append(func(A[:, :, y:y + dims[0], x:x + dims[1]], S))
        results_row.append(torch.vstack(results_cols))
    results_row = torch.vstack(results_row)
    """
    rows = []
    for y in range(results_row.shape[-2]):
        cols = []
        for x in range(results_row.shape[-1]):
            cols.append(results_row[ :, :, y, x])
        rows.append(torch.vstack(cols))
    rows = torch.vstack(rows)
    """
    return results_row


def estimate_gamma_cube(img, S):
    Nb = (img.shape[1] - 1) // 2

    low_lp_d = img[:, :Nb, :, :]
    low_lp = img[:, Nb:2 * Nb, :, :]
    high_lp_d = img[:, 2 * Nb, None, :, :]

    Hd = []

    Hd.append(torch.flatten(low_lp_d, start_dim=2))

    Hd.append(torch.flatten(high_lp_d, start_dim=2))
    Hd = torch.cat(Hd, dim=1).transpose(1, 2)

    Hd_p = torch.adjoint(Hd)
    HHd = torch.matmul(Hd_p, Hd)
    B = torch.linalg.solve(HHd, Hd_p)

    gamma = []
    for k in range(Nb):
        b = low_lp[:, k, :, :]
        bd = low_lp_d[:, k, :, :]
        b = torch.flatten(b, start_dim=1)[:, :, None]
        bd = torch.flatten(bd, start_dim=1)[:, :, None]
        gamma.append(torch.matmul(B, (b - bd)))
    gamma = torch.cat(gamma, -1)[None, :, :, :]
    gamma = pad(gamma, (0, S - Nb, 0, S - Nb - 1))

    return gamma


def compH_injection(img, S):
    Nb = img.shape[1] - 2
    lowres = img[:, :-2, :, :]
    highres = img[:, -2, :, :][:, None, :, :]
    gamma = img[:, -1, :, :]

    # Compute H
    Hlow = torch.flatten(lowres, start_dim=-2)
    Hhigh = torch.flatten(highres, start_dim=-2)

    H = torch.cat([Hlow, Hhigh], dim=1).transpose(1, 2)

    g = gamma[:, :Nb + 1, :Nb]

    mul_Hg = torch.matmul(H, g)
    hlow_en = Hlow + mul_Hg.transpose(1, 2)
    hlow_en = torch.reshape(hlow_en, lowres.shape)
    return hlow_en
