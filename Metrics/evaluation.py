import torch
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch

from . import metrics_rr as rr
from . import metrics_fr as fr


def evaluation_rr(out_lr, ms_lr, ratio, flag_cut=True, dim_cut=11, L=16):

    if flag_cut:
        out_lr = out_lr[:, :, dim_cut-1:-dim_cut, dim_cut-1:-dim_cut]
        ms_lr = ms_lr[:, :, dim_cut-1:-dim_cut, dim_cut-1:-dim_cut]

    out_lr = torch.clip(out_lr, 0, 2 ** L)

    ergas = rr.ERGAS(ratio).to(out_lr.device)
    sam = rr.SAM().to(out_lr.device)
    q2n = rr.Q2n().to(out_lr.device)

    ergas_index, _ = ergas(out_lr, ms_lr)
    sam_index, _ = sam(out_lr, ms_lr)
    q2n_index, _ = q2n(out_lr, ms_lr)

    return ergas_index.item(), sam_index.item(), q2n_index.item()


def evaluation_fr(out, pan, ms_lr, ms, ratio, sensor, overlap):

    if sensor == 'PRISMA':
        starting = 3
    elif sensor == 'Pavia':
        starting = 3
    else:
        starting = 3

    kernel = mtf_kernel_to_torch(gen_mtf(ratio, sensor, nbands=out.shape[1]))

    filter = torch.nn.Conv2d(out.shape[1], out.shape[1], kernel_size=kernel.shape[2], groups=out.shape[1], padding='same', padding_mode='replicate', bias=False)
    filter.weight = torch.nn.Parameter(kernel.type(out.dtype).to(out.device))
    filter.weight.requires_grad = False

    out_lp = filter(out)
    out_lr = out_lp[:, :, starting::ratio, starting::ratio]

    q2n = rr.Q2n().to(out.device)
    d_sr = fr.D_sR().to(out.device)

    # Spectral Assessment
    q2n_index, _ = q2n(out_lr, ms_lr)
    d_lambda_index = 1 - q2n_index

    # Spatial Assessment
    d_sr_index = d_sr(out, pan)

    # Quality with No Reference (QNR) calculation
    qnr = (1 - d_lambda_index) * (1 - d_sr_index)

    return (d_lambda_index.item(),
            d_sr_index.item(),
            qnr.item()
            )