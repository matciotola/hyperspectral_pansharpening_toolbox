import torch
from Utils.spectral_tools import mtf
from torch.nn import functional as func
import math

from Utils.imresize_bicubic import imresize


def SR_D(ordered_dict, ts=7, ol=4, n_atoms=10):

    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio
    sensor = ordered_dict.sensor

    # equalization

    pan = pan.repeat(1, ms.shape[1], 1, 1)

    hr_shapes = pan.shape

    pan = (pan - pan.mean(dim=(2, 3), keepdim=True)) / pan.std(dim=(2, 3), keepdim=True) * ms.std(dim=(2, 3), keepdim=True) + ms.mean(dim=(2, 3), keepdim=True)

    # extract details using mtf-based filters

    ms_lp = mtf(ms, sensor, ratio)
    ms_d = ms - ms_lp

    pan_lp = mtf(pan, sensor, ratio)
    pan_lp = imresize(func.interpolate(pan_lp, scale_factor=1/ratio, mode='nearest-exact'), scale=ratio)
    pan_d = pan - pan_lp

    # decimation ms

    ms_lr = func.interpolate(ms_d, scale_factor=1/ratio, mode='nearest-exact')

    # decimation pan

    pan_lr = func.interpolate(mtf(pan_d, sensor, ratio), scale_factor=1/ratio, mode='nearest-exact')

    # dictionary learning
    dh, dl, y_tilde_k = dictionary_learning(pan_d, pan_lr, ms_lr, ratio, ts, ol)

    # sparse coefficient estimation and hr signal reconstruction
    residual = omp_rec_detile(dl, dh, y_tilde_k, hr_shapes, ratio, ol, ts, n_atoms)

    fused = ms + residual


    return fused


def dictionary_learning(pan_d, pan_lr_d, ms_lr_d, resize_factor, ts, ol):

    bs = pan_d.shape[0]
    nr = math.ceil((pan_d.shape[2] / resize_factor - ol) / (ts - ol))
    nc = math.ceil((pan_d.shape[3] / resize_factor - ol) / (ts - ol))
    nbands = pan_d.shape[1]

    dh = torch.zeros((bs, ts ** 2 * resize_factor ** 2 * nbands, nr*nc), dtype=pan_d.dtype, device=pan_d.device)
    dl = torch.zeros((bs, ts ** 2 * nbands, nr*nc), dtype=pan_d.dtype, device=pan_d.device)
    y_tilde_k = torch.zeros((bs, ts ** 2 * nbands, nr*nc), dtype=pan_d.dtype, device=pan_d.device)

    i_count = 0

    for irow in range(nr):
        for icol in range(nc):

            shift_r = 0
            shift_c = 0
            if (irow == nr - 1) and ((ms_lr_d.shape[2] - ol) % (ts - ol) != 0):
                shift_r = ts - ol - (ms_lr_d.shape[2] - ol) % (ts - ol)
            if (icol == nc - 1) and ((ms_lr_d.shape[3] - ol) % (ts - ol) != 0):
                shift_c = ts - ol - (ms_lr_d.shape[3] - ol) % (ts - ol)

            block_r_ini, block_r_end = irow * (ts - ol) * resize_factor - shift_r * resize_factor, ((irow + 1) * ts - irow * ol) * resize_factor - shift_r * resize_factor
            block_c_ini, block_c_end = icol * (ts - ol) * resize_factor - shift_c * resize_factor, ((icol + 1) * ts - icol * ol) * resize_factor - shift_c * resize_factor

            block_rl_ini, block_rl_end = irow * (ts - ol) - shift_r, (irow + 1) * ts - irow * ol - shift_r
            block_cl_ini, block_cl_end = icol * (ts - ol) - shift_c, (icol + 1) * ts - icol * ol - shift_c

            for iband in range(nbands):
                colmn = pan_d[:, iband, block_r_ini:block_r_end, block_c_ini:block_c_end]
                colmn_lr = pan_lr_d[:, iband, block_rl_ini:block_rl_end, block_cl_ini:block_cl_end]
                colmn_y = ms_lr_d[:, iband, block_rl_ini:block_rl_end, block_cl_ini:block_cl_end]
                dh[:, iband * ts ** 2 * resize_factor ** 2:iband * ts ** 2 * resize_factor ** 2 + colmn.flatten(1).shape[1], i_count] = colmn.transpose(1, 2).flatten(1)
                dl[:, iband * ts ** 2:iband * ts ** 2 + colmn_lr.flatten(1).shape[1], i_count] = colmn_lr.transpose(1, 2).flatten(1)
                y_tilde_k[:, iband * ts ** 2:iband * ts ** 2 + colmn_y.flatten(1).shape[1], i_count] = colmn_y.transpose(1, 2).flatten(1)

            i_count += 1

    return dh, dl, y_tilde_k


def omp_rec_detile(dl, dh, y_tilde_k, hr_shapes, resize_factor, ol, ts, n_atoms):

    bs, c_ms, h_pan, l_pan = hr_shapes

    residual = torch.zeros(hr_shapes, dtype=dl.dtype, device=dl.device)
    countpx = torch.zeros(hr_shapes, dtype=dl.dtype, device=dl.device)

    nr = math.ceil((h_pan / resize_factor - ol) / (ts - ol))
    nc = math.ceil((l_pan / resize_factor - ol) / (ts - ol))

    shift_r_glob = 0
    shift_c_glob = 0

    if (h_pan / resize_factor - ol) % (ts - ol) != 0:
        shift_r_glob = int(ts - ol - (h_pan / resize_factor - ol) % (ts - ol))

    if (l_pan / resize_factor - ol) % (ts - ol) != 0:
        shift_c_glob = int(ts - ol - (l_pan / resize_factor - ol) % (ts - ol))

    alpha_count = 0
    latom = dl.shape[2]
    dict_sizer = y_tilde_k.shape[2]
    iatom = 0

    for irow in range(nr):
        for icol in range(nc):
            if irow == nr - 1:
                shift_r = shift_r_glob
            else:
                shift_r = 0
            if icol == nc - 1:
                shift_c = shift_c_glob
            else:
                shift_c = 0
            blockr_ini, blockr_end = irow * (ts - ol) * resize_factor - shift_r * resize_factor, ((irow + 1) * ts - irow * ol) * resize_factor - shift_r * resize_factor
            blockc_ini, blockc_end = icol * (ts - ol) * resize_factor - shift_c * resize_factor, ((icol + 1) * ts - icol * ol) * resize_factor - shift_c * resize_factor

            lr = blockr_end - blockr_ini
            lc = blockc_end - blockc_ini

            y_cur = y_tilde_k[:, :, iatom:iatom+1]

            # sparse coding with OMP for MS data
            alpha, inds = omp(dl, y_cur, c_ms, iatom, n_atoms)

            for iband in range(c_ms):
                reconstructed_patch = torch.matmul(dh[:, iband * ts ** 2 * resize_factor ** 2:(iband + 1) * ts ** 2 * resize_factor ** 2, inds], alpha[:, :, iband:iband+1])
                residual[:, iband, blockr_ini:blockr_end, blockc_ini:blockc_end] += reconstructed_patch.reshape((bs, lc, lr)).transpose(1, 2)
                countpx[:, iband, blockr_ini:blockr_end, blockc_ini:blockc_end] += 1

            alpha_count += 1

    residual = residual / countpx

    return residual


def omp(d, y, nbands, iatom, n_atoms):
    bs, l_atom_x, l_atom_y = d.shape
    n_x = round(l_atom_x / nbands)
    n_y = round(l_atom_y / nbands)

    res = torch.clone(y)
    delta = 0
    curr_delta = torch.sum(res ** 2, dim=1, keepdim=True)
    j = 0
    indx = []
    while curr_delta > delta and j < n_atoms:
        j += 1
        if j == 1:
             indx.append(iatom)
        else:
            proj = torch.matmul(d.transpose(1, 2), res)
            imax = torch.argmax(torch.abs(proj)).item()
            indx.append(imax)

        a = torch.zeros((bs, j, nbands), dtype=d.dtype, device=d.device)
        da = []
        for iband in range(nbands):
            di = d[:, iband*n_x:(iband+1)*n_x, indx[:j]]
            yi = y[:, iband*n_x:(iband+1)*n_x, :]
            dit_di = torch.matmul(di.transpose(1, 2), di)

            if torch.det(dit_di) > 0.1:
                a[:, :, iband:iband+1] = torch.matmul(torch.inverse(dit_di), torch.matmul(di.transpose(1, 2), yi))
            da.append(torch.matmul(di, a[:, :, iband:iband+1]))
        da = torch.cat(da, 1)
        res = y - da
        curr_delta = torch.sum(res ** 2, dim=1, keepdim=True)

    return a, indx



if __name__ == '__main__':
    import numpy as np
    from scipy import io
    from Utils.interpolator_tools import interp23tap
    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_1_zoom.mat')

    ratio = 4
    sensor = 'WV3'

    ms = torch.from_numpy(interp23tap(temp['I_MS_LR'], ratio).astype(np.float64)).permute(2, 0, 1).unsqueeze(0).double()
    pan = torch.from_numpy(temp['I_PAN'].astype(np.float64)).unsqueeze(0).double()


    fused = SR_D(ms, pan, ratio, sensor)
