import torch
from Utils.pansharpening_aux_tools import mldivide
from .vca import vca

def HySURE(ordered_dict, intersection=None):

    ms_lr = torch.clone(ordered_dict.ms_lr)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio
    bs, _, nl, nc = pan.shape
    bs, nb, nlh, nch = ms_lr.shape

    if intersection is None:
        intersection = torch.arange(0, ms_lr.shape[1]).long()
    contigous = torch.clone(intersection)

    # Regularization parameters
    lambda_r = 10
    lambda_b = 10
    # For the denoising with SVD, we need to specify the number of bands we
    # want to keep
    p = 10

    # other hyperparameters
    basis_type = 'VCA'
    lambda_phi = 1e-2
    lambda_m = 1

    # Normalize the data
    max_ms = torch.max(ms_lr)
    ms_lr = ms_lr / max_ms
    pan = pan / max_ms

    # Estimate the sensor response
    hsize_h, hsize_w = 10, 10
    shift = 1
    blur_center = 0

    v, r_est, b_est = sensor_response_estimation(ms_lr, pan, ratio, intersection, contigous, p, lambda_r, lambda_b, hsize_h, hsize_w, shift, blur_center)

    # 3. Data Fusion
    z_im_hat = data_fusion(ms_lr, pan, ratio, r_est, b_est, p, basis_type, lambda_phi, lambda_m)

    # Denoise the data with V
    z_hat = im2mat(z_im_hat)
    z_hat_denoised = torch.bmm(v, torch.bmm(v.transpose(-2, -1), z_hat))
    fused = mat2im(z_hat_denoised, nl, nc, nb)

    return fused * max_ms

def im2mat(img):
    return torch.flatten(img.transpose(2, 3), start_dim=2, end_dim=3)


def mat2im(img, nl, nc, nb):
    return torch.reshape(img, (img.shape[0], img.shape[1], nc, nl)).transpose(2, 3)


def conv_c(img, kernel, nl, nc, nb):
    X = im2mat(torch.real(torch.fft.ifft2(torch.fft.fft2(torch.reshape(img, (img.shape[0], img.shape[1], nc, nl)).transpose(2, 3)) * kernel.repeat(1, img.shape[1], 1, 1))))
    return X


def downsample_hs(img, downsample_factor, shift):

    return img[:, :, shift::downsample_factor, shift::downsample_factor]


def upsample_hs(img, downsample_factor, nl, nc, nb, shift):

    bs, nbh, nlh, nch = img.shape
    aux = torch.zeros((bs, nbh, nlh * downsample_factor, nch * downsample_factor), dtype=img.dtype, device=img.device)
    aux[:, :, shift::downsample_factor, shift::downsample_factor] = img
    return aux[:, :, :nl, :nc]

def toeplitz(c, r=None):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j - i].reshape(*shape)


def sensor_response_estimation(ms, pan, downsample_factor, intersection, contigous, p, lambda_r, lambda_b, hsize_h, hsize_w, shift, blur_center):

    # kernel support: [lx*2+1, ly*2+1]
    lx = 4
    ly = 4

    # Blur the PAN with a strong blur
    _, nb, nl, nc = pan.shape

    # Blur operator
    middle_l = nl // 2
    middle_c = nc // 2

    # Blur matrix
    blur_matrix_h = torch.zeros((nl, nc), dtype=ms.dtype, device=ms.device)
    blur_matrix_h[middle_l - lx:middle_l + lx + 1, middle_c - ly:middle_c + ly + 1] = 1
    blur_matrix_h = blur_matrix_h / torch.sum(blur_matrix_h)

    # Circularly shift
    blur_matrix_h = torch.fft.ifftshift(blur_matrix_h)
    blur_matrix_h = blur_matrix_h / torch.sum(blur_matrix_h)

    f_blur_matrix = torch.fft.fft2(blur_matrix_h)

    pan_m = im2mat(pan)
    pan_b = conv_c(pan_m, f_blur_matrix, nl, nc, nb)

    pan_b_im = mat2im(pan_b, nl, nc, nb)
    pan_b_im_down = downsample_hs(pan_b_im, downsample_factor, shift)

    pan_b_down = im2mat(pan_b_im_down)

    # Blur the MS with a corresponding scaled blur

    bs, l, nlh, nch = ms.shape

    # Correspondingly scaled blur's support
    lx = round(lx / downsample_factor)
    ly = round(ly / downsample_factor)

    # Blur operator
    middle_lh = round(nlh / 2)
    middle_ch = round(nch / 2)

    # Blur matrix
    blur_matrix_h = torch.zeros((nlh, nch), dtype=ms.dtype, device=ms.device)
    blur_matrix_h[middle_lh - lx:middle_lh + lx + 1, middle_ch - ly:middle_ch + ly + 1] = 1
    blur_matrix_h = blur_matrix_h / torch.sum(blur_matrix_h)

    # Circularly shift
    blur_matrix_h = torch.fft.ifftshift(blur_matrix_h)

    # Normalize
    blur_matrix_h = blur_matrix_h / torch.sum(blur_matrix_h)

    fb_h = torch.fft.fft2(blur_matrix_h)

    # Blur MS
    ms_m = im2mat(ms)
    ms_b = conv_c(ms_m, fb_h, nlh, nch, nb)

    # Estimate R on the blurred data

    R = torch.zeros((l), dtype=ms.dtype, device=ms.device)

    no_hs_bands = len(intersection)
    col_aux = torch.zeros((no_hs_bands - 1), dtype=ms.dtype, device=ms.device)
    col_aux[0] = 1

    row_aux = torch.zeros((no_hs_bands), dtype=ms.dtype, device=ms.device)
    row_aux[0] = 1
    row_aux[1] = -1

    d = toeplitz(col_aux, row_aux)

    if torch.sum(torch.diff(contigous) != 1):
        d[torch.diff(contigous) != 1, :] = 0

    ddt = d.T @ d
    to_inv = torch.bmm(ms_b[:, intersection, :], ms_b[:, intersection, :].transpose(-1, -2)) + lambda_r * ddt[None, :, :].repeat(bs, 1, 1)
    r = mldivide(torch.bmm(ms_b[:, intersection, :], pan_b_down.transpose(-2, -1)), to_inv)
    R = r.transpose(-2, -1)

    # Data denoising

    mask = torch.zeros((nl, nc), dtype=ms.dtype, device=ms.device)
    mask[shift::downsample_factor, shift::downsample_factor] = 1
    mask_im = mask[None, None, :, :].repeat(bs, p, 1, 1)

    ms_up = upsample_hs(ms, downsample_factor, nl, nc, nb, shift)

    ms_h_up = im2mat(ms_up)

    V, _, _ = torch.linalg.svd(ms_m)
    V = V[:, :, :p]


    ms_h_up = torch.bmm(torch.bmm(V, V.transpose(-2, -1)), ms_h_up)

    # Estimate B on the original observed data

    col_aux = torch.zeros((hsize_w - 1), dtype=ms.dtype, device=ms.device)
    col_aux[0] = 1

    row_aux = torch.zeros((hsize_h), dtype=ms.dtype, device=ms.device)
    row_aux[0] = 1
    row_aux[1] = -1

    dv = toeplitz(col_aux, row_aux)
    dh = dv.T

    # BTTB matrix

    a_h = torch.kron(dh.T, torch.eye(hsize_w))

    at_ah = a_h.T @ a_h

    av = torch.kron(torch.eye(hsize_h), dv)
    at_av = av.T @ av

    ryh = torch.bmm(R, ms_h_up)

    rym_img = mat2im(ryh, nl, nc, nb)
    ymymt = torch.zeros((hsize_h*hsize_w, hsize_h*hsize_w), dtype=ms.dtype, device=ms.device)
    rtyhymt = torch.zeros((bs, 1, hsize_h*hsize_w), dtype=ms.dtype, device=ms.device)

    for idx_h in range((hsize_h - 1) // 2, nl - ((hsize_h - 1) // 2) - 1):
        for idx_w in range((hsize_w - 1) // 2, nc - ((hsize_w - 1) // 2) - 1):
            if mask_im[0, 0, idx_h, idx_w] == 1:
                if hsize_h % 2 == 0 and hsize_w % 2 == 0:
                    patch = pan[:, :, idx_h - ((hsize_h - 1) // 2):idx_h + ((hsize_h) // 2) + 1, idx_w - ((hsize_w - 1) // 2):idx_w + (hsize_w // 2) + 1]
                elif hsize_h % 2 != 0 and hsize_w % 2 == 0:
                    patch = pan[:, :, idx_h - (hsize_h - 1) // 2:idx_h + ((hsize_h) // 2), idx_w - ((hsize_w - 1) // 2):idx_w + ((hsize_w) // 2) + 1]
                elif hsize_h % 2 == 0 and hsize_w % 2 != 0:
                    patch = pan[:, :, idx_h - (hsize_h - 1) // 2:idx_h + ((hsize_h) // 2) + 1, idx_w - ((hsize_w - 1) // 2):idx_w + ((hsize_w) // 2)]
                else:
                    patch = pan[:, :, idx_h - (hsize_h - 1) // 2:idx_h + ((hsize_h) // 2), idx_w - ((hsize_w - 1) // 2):idx_w + ((hsize_w) // 2)]

                ymymt = ymymt + torch.bmm(torch.flatten(patch.transpose(2, 3), start_dim=2, end_dim=3).transpose(-2, -1), torch.flatten(patch.transpose(2, 3), start_dim=2, end_dim=3))
                rtyhymt = rtyhymt + torch.squeeze(rym_img[:, :, idx_h, idx_w]) * torch.flatten(patch.transpose(2, 3), start_dim=2, end_dim=3)

    b_vec = mldivide(rtyhymt.transpose(-2,-1), ymymt + lambda_b * (at_ah + at_av)[None, :, :])
    b_vecim = mat2im(b_vec.transpose(-1,-2), hsize_w, hsize_w, b_vec.shape[1])

    B = torch.zeros((bs, nb, nl, nc), dtype=ms.dtype, device=ms.device)

    if (hsize_h % 2 == 0) and (hsize_w % 2) == 0:
        B[:, :, middle_l - ((hsize_h - 1) // 2) - blur_center:middle_l + ((hsize_h) // 2) + 1 - blur_center, middle_c - ((hsize_w - 1) // 2) - blur_center:middle_c + ((hsize_w) // 2) + 1 - blur_center] = b_vecim
    elif (hsize_h % 2 != 0) and (hsize_w % 2) == 0:
        B[:, :, middle_l - ((hsize_h - 1) // 2) - blur_center:middle_l + ((hsize_h) // 2), middle_c - ((hsize_w - 1) // 2) - blur_center:middle_c + ((hsize_w) // 2) + 1 - blur_center] = b_vecim
    elif (hsize_h % 2 == 0) and (hsize_w % 2) != 0:
        B[:, :, middle_l - ((hsize_h - 1) // 2) - blur_center:middle_l + ((hsize_h) // 2) + 1 - blur_center, middle_c - ((hsize_w - 1) // 2) - blur_center:middle_c + ((hsize_w) // 2)] = b_vecim
    else:
        B[:, :, middle_l - ((hsize_h - 1) // 2) - blur_center:middle_l + ((hsize_h) // 2), middle_c - ((hsize_w - 1) // 2) - blur_center:middle_c + ((hsize_w) // 2)] = b_vecim

    B = torch.fft.ifftshift(B)

    B_vol = torch.sum(B)
    B = B / B_vol
    R = R / B_vol

    return V, R, B


def vector_soft_col_iso(x1, x2, tau):
    nu = torch.sqrt(torch.sum(x1 ** 2, dim=1) + torch.sum(x2 ** 2, dim=1))
    a = torch.clip(nu - tau, 0, torch.inf)
    a = a / (a + tau)
    a = a.repeat(x1.shape[1], 1)
    y1 = a * x1
    y2 = a * x2

    return y1, y2

def data_fusion(y_him, y_mim, downsampling_factor, R, B, p, basis_type, lambda_phi, lambda_m):

    mu = 0.05
    iters = 200

    # 1. Precomputations

    bs, _, nl, nc = y_mim.shape

    dh = torch.zeros((nl, nc), dtype=y_mim.dtype, device=y_mim.device)
    dh[0, 0] = 1
    dh[0, -1] = -1

    dv = torch.zeros((nl, nc), dtype=y_mim.dtype, device=y_mim.device)
    dv[0, 0] = 1
    dv[-1, 0] = -1

    fdh = torch.fft.fft2(dh)
    fdhc = torch.conj(fdh)

    fdv = torch.fft.fft2(dv)
    fdvc = torch.conj(fdv)

    # Fourier transform of B

    fb = torch.fft.fft2(B)
    fbc = torch.conj(fb)

    idb_b = fbc / (torch.abs(fb) ** 2 + torch.abs(fdh) ** 2 + torch.abs(fdv) ** 2 + 1)
    idb_ii = 1 / (torch.abs(fb) ** 2 + torch.abs(fdh) ** 2 + torch.abs(fdv) ** 2 + 1)
    idb_dh = fdhc / (torch.abs(fb) ** 2 + torch.abs(fdh) ** 2 + torch.abs(fdv) ** 2 + 1)
    idb_dv = fdvc / (torch.abs(fb) ** 2 + torch.abs(fdh) ** 2 + torch.abs(fdv) ** 2 + 1)


    shift = 1
    mask = torch.zeros((bs, 1, nl, nc), dtype=y_mim.dtype, device=y_mim.device)
    mask[:, :, shift::downsampling_factor, shift::downsampling_factor] = 1
    mask_im = mask.repeat(1, p, 1, 1)

    mask = im2mat(mask_im)

    y_him_up = upsample_hs(y_him, downsampling_factor, nl, nc, y_him.shape[1], shift)
    y_h_up = im2mat(y_him_up)

    # 2. Subspace learning

    if basis_type == 'VCA':
        max_vol = 0
        vol = torch.zeros((1, 20), dtype=y_mim.dtype, device=y_mim.device)
        for idx_vca in range(20):
            e_aux, _, _ = vca((y_h_up[:, :, mask[0, 0, :] > 0]).squeeze(0), p, True)
            vol[0, idx_vca] = torch.abs(torch.det(e_aux.T @ e_aux))
            if vol[0, idx_vca] > max_vol:
                e = e_aux
                max_vol = vol[0, idx_vca]

    elif basis_type == 'SVD':
        r_y = y_him
        e, _, _ = torch.linalg.svd(r_y)
        e = e[:, :, :p]

    # 3. ADMM/SALSA

    # Auxiliary matrices

    ie = e.T @ e + mu * torch.eye(p)
    yyh = e.T @ y_h_up

    ire = lambda_m * e.T @ (R.transpose(-1,-2) @ R) @ e + mu * torch.eye(p)
    ym = im2mat(y_mim)
    yym = e.T[None, :, :] @ R.transpose(-1,-2) @ ym

    # Initialization
    x = torch.zeros((bs, p, nl*nc), dtype=y_mim.dtype, device=y_mim.device)
    v1 = torch.clone(x)
    d1 = torch.clone(x)
    v2 = torch.clone(x)
    d2 = torch.clone(x)
    v3 = torch.clone(x)
    d3 = torch.clone(x)
    v4 = torch.clone(x)
    d4 = torch.clone(x)

    for i in range(iters):

        x = conv_c(v1 + d1, idb_b, nl, nc, p) + conv_c(v2 + d2, idb_ii, nl, nc, p) + conv_c(v3 + d3, idb_dh, nl, nc, p) + conv_c(v4 + d4, idb_dv, nl, nc, p)
        nu1 = conv_c(x, fb, nl, nc, p) - d1
        v1 = mldivide(mu*nu1 + yyh, ie[None, :, :]) * mask + nu1 * (1 - mask)

        nu2 = x - d2
        v2 = mldivide(mu*nu2 + lambda_m * yym, ire)

        nu3 = conv_c(x, fdh, nl, nc, p) - d3
        nu4 = conv_c(x, fdv, nl, nc, p) - d4

        v3, v4 = vector_soft_col_iso(nu3, nu4, lambda_phi / mu)

        d1 = -nu1 + v1
        d2 = -nu2 + v2
        d3 = -nu3 + v3
        d4 = -nu4 + v4

    z_hat = e[None, :, :] @ x
    z_im_hat = mat2im(z_hat, nl, nc, p)

    return z_im_hat
