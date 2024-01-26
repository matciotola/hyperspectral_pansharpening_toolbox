import torch
import numpy as np
from Utils.pansharpening_aux_tools import mldivide
from Utils.spectral_tools import gen_mtf, fspecial_gauss
import torchvision.transforms.functional as tf
from scipy.interpolate import RectBivariateSpline
import math


def BayesianNaive(ordered_dict):
    ms_lr = ordered_dict.ms_lr
    pan = ordered_dict.pan
    overlap = torch.arange(0, 32).float()  # ordered_dict.overlap
    ratio = ordered_dict.ratio

    blur_kernel = torch.from_numpy(gen_mtf(ratio, ordered_dict.sensor, 15, 1).squeeze()).type(ms_lr.dtype).to(ms_lr.device)
    fused = BayesianMethod(ms_lr, pan, overlap, ratio, blur_kernel, 'Gaussian')

    return fused


def BayesianSparse(ordered_dict):
    ms_lr = ordered_dict.ms_lr
    pan = ordered_dict.pan
    overlap = torch.arange(0, 32).float()  # ordered_dict.overlap
    ratio = ordered_dict.ratio

    blur_kernel = torch.from_numpy(gen_mtf(ratio, ordered_dict.sensor, 15, 1).squeeze()).type(ms_lr.dtype).to(ms_lr.device)

    fused = BayesianMethod(ms_lr, pan, overlap, ratio, blur_kernel, 'Sparse')

    return fused


def BayesianMethod(ms_lr, pan, overlap, ratio, blur_kernel, prior):
    xh = torch.clone(ms_lr)
    xm = torch.clone(pan)

    overlap = torch.clone(overlap)

    l_pan = int(overlap[-1].item()) + 1

    # Reshape the input data

    bs, n_hs, n_dr, n_dc = xh.shape
    _, n_ms, nr, nc = xm.shape

    vxh = xh.transpose(2, 3).flatten(2)
    vxm = xm.transpose(2, 3).flatten(2)

    # Noise variances

    ch_inv = torch.eye(n_hs, dtype=xh.dtype).repeat(bs,1,1)
    cm_inv = torch.eye(n_ms, dtype=xh.dtype).repeat(bs,1,1)

    # Subspace identification

    e_hyper, p_dec, _, l = id_hs_sub(xh)

    size_im = [bs, l, nr, nc]

    # Spectral mixture: spectral response R

    psf_z = [torch.ones(overlap.shape, dtype=xh.dtype) / overlap.shape[-1], torch.zeros((n_hs - l_pan), dtype=xh.dtype)]
    psf_z = torch.cat(psf_z).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)

    psfy_b = kernel_to_matrix(blur_kernel, nr, nc)
    psfy_ds_r = ratio

    # Interpolate the HS image with zeros

    xh_int = torch.zeros((bs, n_hs, nr, nc), dtype=xh.dtype, device=xh.device)
    xh_int[:, :, ::psfy_ds_r, ::psfy_ds_r] = xh
    vxh_int = xh_int.transpose(2, 3).flatten(2)

    # Generation of the interpolated image with spline interpolation

    xd_dec = ima_interp_spline(var_dim(xh, p_dec), psfy_ds_r)

    if prior == 'Sparse':
        d_s = dic_para(xm, e_hyper)
        n_ao = 10
    else:
        d_s = None
        n_ao = 1

    vx = sylvester_fusion(vxh_int, vxm, psfy_b, psfy_ds_r, psf_z, ch_inv, cm_inv, e_hyper, xd_dec, size_im, n_ao, prior, d_s)

    vx_bayesian_fusion = torch.matmul(e_hyper, vx)
    fused = torch.reshape(vx_bayesian_fusion, (bs, n_hs, nc, nr)).transpose(2, 3)

    return fused


def sylvester_fusion(vxh_int, vxm, psfy_b, psfy_ds_r, psf_z, ch_inv, cm_inv, e_hyper, xd_dec, size_im, n_ao, prior, d_s=None):

    bs, l, nr, nc = size_im

    n_dr = nr // psfy_ds_r
    n_dc = nc // psfy_ds_r

    e_multi = torch.matmul(psf_z, e_hyper)

    # Auxiliary matrices

    fbm = torch.fft.fft2(psfy_b)
    fbm_c = torch.conj(fbm)
    fbs = fbm[None, None, :, :].repeat(bs, l, 1, 1)
    fbc_s = (fbm_c)[None, None, :, :].repeat(bs, l, 1, 1)
    b2sum = pplus(torch.abs(fbs) ** 2 / psfy_ds_r ** 2, n_dr, n_dc)

    # Normalize HS and PAN data
    nm_vxh_int = torch.matmul(ch_inv, vxh_int)
    nm_vxm = torch.matmul(cm_inv, vxm)
    yyh = torch.matmul(e_hyper.transpose(-1, -2), nm_vxh_int)
    yym = torch.matmul(e_multi.transpose(-1, -2), nm_vxm)
    er1e = torch.matmul(e_hyper.transpose(-1, -2), torch.matmul(ch_inv, e_hyper))
    rer2re = torch.matmul(e_multi.transpose(-1, -2), torch.matmul(cm_inv, e_multi))

    inv_cov = 1e-3 * torch.eye(l, dtype=yyh.dtype).repeat(bs, 1, 1)  # Sparse?

    q, cc, inv_di, inv_lbd, c2cons = faster_aux_mat(yyh, yym, fbc_s, size_im, n_dr, n_dc, inv_cov, er1e, rer2re, b2sum)

    for i in range(n_ao):
        mean_vim = torch.fft.fft2(xd_dec).transpose(2, 3).flatten(2)
        vx_fft = faster_fusion(mean_vim, q, cc, inv_di, inv_lbd, c2cons, fbs, fbc_s, size_im, n_dr, n_dc, psfy_ds_r)
        vx = torch.fft.ifft2(torch.reshape(vx_fft, (size_im[0], size_im[1], size_im[3], size_im[2])).transpose(2, 3)).real.transpose(2,3).flatten(2)
        if prior == 'Sparse':
            for k in range(l):
                xd_dec[:, k, :, :] = restore_from_supp(torch.reshape(vx[:, k:k+1, :], (bs, 1, nc, nr)).transpose(2,3), nr, nc, d_s)

    return vx


def restore_from_supp(y, nr, nc, d_s):
    m = d_s.shape[1]
    patsize = int(math.sqrt(m))

    py = torch.nn.functional.unfold(y.transpose(2,3), kernel_size=patsize, padding=0, stride=1)
    num_patches = py.shape[2]
    py_hat = torch.zeros(py.shape, dtype=py.dtype, device=py.device)

    for i in range(num_patches):
        py_hat[:, :, i:i+1] = torch.matmul(d_s[:, :, :, i], py[:, :, i:i+1])

    blockx = patsize
    blocky = patsize
    final_numestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
    final_extentestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)

    for index_i in range(blocky):
        for index_j in range(blockx):
            tempesti = torch.reshape(py_hat[:, index_i * blockx + index_j, :].transpose(-2, -1), (
            y.shape[0], y.shape[1], y.shape[3] - blocky + 1, y.shape[2] - blockx + 1)).transpose(2, 3)
            numestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
            extentestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
            extentestimate[:, :, :tempesti.shape[2], :tempesti.shape[3]] = tempesti
            numestimate[:, :, :tempesti.shape[2], :tempesti.shape[3]] = 1

            extentestimate = torch.roll(extentestimate, (index_j - 1, index_i - 1), dims=(2, 3))
            numestimate = torch.roll(numestimate, (index_j - 1, index_i - 1), dims=(2, 3))

            final_numestimate = final_numestimate + numestimate
            final_extentestimate = final_extentestimate + extentestimate

    x_hat = final_extentestimate / final_numestimate

    return x_hat


def soft(x, t):

    if torch.sum(torch.abs(t)) == 0:
        y = x
    else:
        y = torch.clip(torch.abs(x) - t, min = 0)
        y = y / (y + t) * x

    return y


def sunsal_simple(m, y, lambda_, err):

    bs, p, lm = m.shape
    _, l, n = y.shape

    assert lm == l, 'mixing matrix M and data set y are inconsistent'

    # Set the defaults for the optimization parameters

    # Maximum number of iterations
    al_iters = 100
    # tolerance for the primal and dual residues
    tol = err

    # Local variables

    norm_y = torch.sqrt(torch.mean(torch.abs(y) ** 2, dim=(1,2)))

    m = m / norm_y
    y = y / norm_y
    lambda_ = lambda_ / (norm_y ** 2)

    # Constants and initialization

    mu_al = 0.01
    mu = 10 * torch.mean(lambda_) + mu_al

    uf, sf, _ = torch.linalg.svd(torch.matmul(m, m.transpose(1, 2)))
    iff = torch.matmul(torch.matmul(uf, torch.diag_embed(1 / (sf + mu), dim1=-2, dim2=-1)), uf.transpose(-1, -2))

    yy = torch.matmul(m, y)

    # initialization

    # no initial solution supplied
    x = torch.matmul(torch.matmul(iff, m), y)
    z = torch.clone(x)
    d = z * 0

    # al iterations - main body

    mu_changed = 0

    for i in range(al_iters):
        if i % 10 == 0:
            z0 = torch.clone(z)

        # minimize with respect to z
        z = torch.clip(soft(x-d, lambda_/mu), min=0)
        x = torch.matmul(iff, yy + mu * (z + d))

        # lagrange multiplier update
        d = d - (x - z)

        if i % 10 == 0:
            res_p = torch.norm(torch.abs(x-z), 'fro')
            res_d = mu * torch.norm(torch.abs(z-z0), 'fro')

            if res_p > 10 * res_d:
                mu = 2 * mu
                d = d / 2
                mu_changed = 1
            elif res_d > 10 * res_p:
                mu = mu / 2
                d = d * 2
                mu_changed = 1

            if mu_changed:
                iff = torch.matmul(torch.matmul(uf, torch.diag_embed(1 / (sf + mu), dim1=-2, dim2=-1)), uf.transpose(-1, -2))
                mu_changed = 0

    return z


def online_dic_learn(alphat, xi, d, a, b, ite):

    bs, m, patch_num = xi.shape
    k = alphat.shape[1]

    rho = 2
    beta = (1 - 1/(ite + 1)) ** rho

    a_temp = torch.zeros((bs, k, k), dtype=xi.dtype, device=xi.device)
    b_temp = torch.zeros((bs, k, m), dtype=xi.dtype, device=xi.device)

    for tt in range(patch_num):
        a_temp = a_temp + torch.matmul(alphat[:, :, tt:tt+1], alphat[:, :, tt:tt+1].transpose(1,2))
        b_temp = b_temp + torch.matmul(xi[:, :, tt:tt+1], alphat[:, :, tt:tt+1].transpose(1, 2)).transpose(1,2)

    a = beta * a + a_temp
    b = beta * b + b_temp

    for j in range(k):
        if a[0, j, j] != 0:
            uj = b[:, j:j+1, :] - torch.matmul(d.transpose(1, 2), a[:, :, j:j+1]).transpose(1,2) / a[:, j:j+1, j:j+1] + d[:, j:j+1, :]
            dj = uj / torch.maximum(torch.norm(uj, dim=(1, 2), keepdim=True), torch.ones(torch.norm(uj, dim=(1,2), keepdim=True).shape))
            d[:, j:j+1, :] = dj

    d = torch.abs(d)
    return d, a, b



def dic_learning_m(x, k, t, patch_num, lambda_, err):

    t0 = 1e-3
    m = x.shape[1]

    a = t0 * torch.eye(k, dtype=x.dtype).repeat(x.shape[0], 1, 1)

    dct = torch.zeros((x.shape[0], k, m), dtype=x.dtype, device=x.device)

    for ii in range(k):
        v = torch.cos(torch.arange(0, m, 1) * ii * np.pi / k)
        if ii > 0:
            v = v - torch.mean(v)
        dct[:, ii, :] = v / torch.norm(v)

    d = dct
    d = d / torch.sqrt(torch.sum(torch.abs(d) ** 2, dim=-1, keepdim=True))

    b = t0 * d

    x = x / torch.sqrt(torch.sum(x * torch.conj(x), dim=1, keepdim=True))

    sele = torch.randint(0, x.shape[-1], (1, t * patch_num)).squeeze()

    for ite in range(t):
        index = sele[ite * patch_num:(ite + 1) * patch_num]
        xi = x[:, :, index]

        alphat = sunsal_simple(d, xi, lambda_, err)

        alphat[torch.abs(alphat) < 1e-4] = 0

        d, a, b = online_dic_learn(alphat, xi, d, a, b, ite)

    d = d / torch.sqrt(torch.sum(torch.abs(d) ** 2, dim=-1, keepdim=True))

    return d, a, b


def dic_learn(xp, patsize):

    pxp = torch.nn.functional.unfold(xp.transpose(2,3), kernel_size=patsize, padding=0, stride=1)

    k = 256
    t = 500
    patch_num = 64
    lambda_ = 0.11
    err = 1e-2

    d, _, _ = dic_learning_m(pxp, k, t, patch_num, lambda_, err)
    ima_dl = xp

    return d, ima_dl


def omp_c(d, x, l, err):
    bs, k, m = d.shape
    n = x.shape[-1]

    a = torch.zeros((bs, k, n), dtype=d.dtype, device=d.device)

    for index in range(n):
        x_i = x[:, :, index:index+1]
        r_i = x_i
        err_iter = torch.sum(torch.abs(r_i) ** 2, dim=1, keepdim=True)

        number_l = 0
        index_select = []

        while number_l < l and err_iter > err:
            derr = torch.matmul(d, r_i)

            max_value, max_index = torch.sort(torch.abs(derr), dim=1, descending=True)
            index_select.append(max_index[:, 0:1])
            number_l = number_l + 1
            a_i = torch.matmul(torch.linalg.pinv(d[:, index_select, :].transpose(1,2)), x_i)
            r_i = x_i - torch.matmul(d[:, index_select, :].transpose(1,2), a_i)
            err_iter = torch.sum(torch.abs(r_i) ** 2, dim=1, keepdim=True)

            if index_select:
                a[:, index_select, index:index+1] = a_i


    return a

def comp_code(y, d, max_atoms, delta):

    bs, k, m = d.shape
    patsize = int(math.sqrt(m))

    py = torch.nn.functional.unfold(y.transpose(2, 3), kernel_size=patsize, padding=0, stride=1)

    l = max_atoms
    err = delta
    alpha = omp_c(d, py, l, err)

    py_hat = torch.matmul(d.transpose(1, 2), alpha)

    blockx = patsize
    blocky = patsize
    final_numestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
    final_extentestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)

    for index_i in range(blocky):
        for index_j in range(blockx):
            tempesti = torch.reshape(py_hat[:, index_i * blockx + index_j, :].transpose(-2, -1), (y.shape[0], y.shape[1], y.shape[3] - blocky + 1, y.shape[2] - blockx + 1)).transpose(2, 3)
            numestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
            extentestimate = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
            extentestimate[:, :, :tempesti.shape[2], :tempesti.shape[3]] = tempesti
            numestimate[:, :, :tempesti.shape[2], :tempesti.shape[3]] = 1

            extentestimate = torch.roll(extentestimate, (index_j - 1, index_i - 1), dims=(2, 3))
            numestimate = torch.roll(numestimate, (index_j - 1, index_i - 1), dims=(2, 3))

            final_numestimate = final_numestimate + numestimate
            final_extentestimate = final_extentestimate + extentestimate

    x_hat = final_extentestimate / final_numestimate

    return x_hat, alpha


def dic_para(x_source, p_inc):

    nb_sub = p_inc.shape[1]
    patsize = 6

    max_atoms = 4
    delta = 1e-3

    d, _ = dic_learn(x_source, patsize)

    ima_dl = x_source

    x_hat, alpha = comp_code(ima_dl, d, max_atoms, delta)
    supp = alpha != 0

    d_s = torch.zeros((x_source.shape[0], patsize**2, patsize**2, alpha.shape[2]), dtype=x_source.dtype, device=x_source.device)

    for j in range(alpha.shape[2]):
        d_aux = d[:, supp[:, :, j].squeeze(), :]
        d_s[:, :, :,  j] = torch.matmul(d_aux.transpose(-1, -2), mldivide(torch.eye(patsize ** 2)[None, :, :].repeat(d_aux.shape[0], 1, 1), d_aux.transpose(-1, -2)))

    return d_s


def faster_fusion(mu, q, cc, inv_di, inv_lbd, c5cons, fbs, fbcs, size_im, n_dr, n_dc, dsf):
    c5bar = c5cons + torch.reshape(torch.matmul(cc.type(mu.dtype), mu), (size_im[0], size_im[1], size_im[3], size_im[2])).transpose(2,3) * inv_lbd
    temp = pplus_s(c5bar / (dsf ** 2) * fbs, n_dr, n_dc)
    inv_quf = c5bar - (temp * inv_di).repeat(1, 1, dsf, dsf) * fbcs
    vxf = torch.matmul(q.type(inv_quf.dtype), inv_quf.transpose(2, 3).flatten(2))

    return vxf


def pplus_s(x, n_dr, n_dc):
    bs, nb, nr, nc = x.shape

    # Sum according to the columns
    temp = torch.reshape(x.transpose(2,3), (bs, nb, nc//n_dc, nr*n_dc)).transpose(2, 3)
    temp = torch.sum(temp, dim=3, keepdim=True)

    # Sum according to the rows
    temp1 = torch.reshape(temp.transpose(2, 3), (bs, nb, n_dc, nr)).transpose(2, 3)
    temp1 = torch.reshape(temp1, (bs, nb, nr // n_dr, n_dc*n_dr)).transpose(2, 3)
    y = torch.reshape(torch.sum(temp1, dim=3, keepdim=True).transpose(2, 3), (bs, nb, n_dr, n_dc)).transpose(2, 3).permute(0, 1, 3, 2)
    return y


def faster_aux_mat(yyh, yym, fbcs, size_im, n_dr, n_dc, inv_cov, er1e, rer2re, b2sum):
    lambda_, q = torch.linalg.eig(mldivide(rer2re + inv_cov, er1e))
    q = q.real
    lambda_ = lambda_.real.squeeze(1)

    cc = mldivide(inv_cov, torch.matmul(er1e, q).squeeze(1))
    temp = mldivide(torch.eye(size_im[1], dtype=er1e.dtype).repeat(er1e.shape[0], 1, 1), torch.matmul(er1e, q).squeeze(1))

    inv_di = 1 / (b2sum[:, :, :n_dr, :n_dc] + lambda_[:, :, None, None].repeat(1, 1, n_dr, n_dc))
    inv_ldb = 1 / (lambda_[:, :, None, None].repeat(1, 1, size_im[2], size_im[3]))
    c3cons = (torch.fft.fft2(torch.reshape(torch.matmul(temp, yyh), (size_im[0], size_im[1], size_im[3], size_im[2])).transpose(2, 3)) * fbcs + torch.fft.fft2(torch.reshape(torch.matmul(temp, yym), (size_im[0], size_im[1], size_im[3], size_im[2])).transpose(2, 3))) * inv_ldb
    return q, cc, inv_di, inv_ldb, c3cons


def pplus(x, n_dr, n_dc):
    bs, nb, nr, nc = x.shape

    # Sum according to the columns
    temp = torch.reshape(x.transpose(2,3), (bs, nb, nc//n_dc, nr*n_dc)).transpose(2,3)
    temp = torch.sum(temp, dim=3, keepdim=True)

    # Sum according to the rows
    temp1 = torch.reshape(temp.transpose(2, 3), (bs, nb, n_dc, nr)).transpose(2, 3)
    temp1 = torch.reshape(temp1, (bs, nb, nr // n_dr, n_dc*n_dr)).transpose(2, 3)
    temp2 = torch.reshape(torch.sum(temp1, dim=3, keepdim=True).transpose(2, 3), (bs, nb, n_dr, n_dc)).transpose(2, 3).permute(0, 1, 3, 2)
    x[:, :, :n_dr, :n_dc] = temp2

    return x
def var_dim(x, p):
    temp = torch.matmul(x.transpose(2, 3).flatten(2).transpose(1, 2), p.transpose(1, 2))
    out = torch.reshape(temp.transpose(-1, -2), (x.shape[0], p.shape[1], x.shape[3], x.shape[2])).transpose(2,3)
    return out

def ima_interp_spline(x, ds_r):

    x = tf.pad(x, 1, padding_mode='symmetric')
    nr = x.shape[2]
    nc = x.shape[3]

    x_interp = []
    for i in range(x.shape[0]):
        temp_interp = []
        for j in range(x.shape[1]):
            f = RectBivariateSpline(np.arange(0, nr, 1), np.arange(0, nc, 1), x[i, j, :, :].numpy())
            xgrid, ygrid = np.meshgrid(np.arange(0, nr, 1 / ds_r), np.arange(0, nc, 1 / ds_r), indexing="ij")
            temp_interp.append(torch.tensor(f.ev(xgrid, ygrid)))
        temp_interp = torch.stack(temp_interp)
        x_interp.append(temp_interp)
    x_interp = torch.stack(x_interp)

    x_interp = x_interp[:, :, ds_r:-ds_r, ds_r:-ds_r]

    return x_interp


def kernel_to_matrix(kernel, nr, nc):

    mid_col = round((nc + 1) / 2)
    mid_row = round((nr + 1) / 2)

    len_hor, len_ver = kernel.shape

    lx = (len_hor - 1) // 2
    ly = (len_ver - 1) // 2

    b = torch.zeros(nr, nc, dtype=kernel.dtype)

    b[mid_row - lx - 1:mid_row + lx, mid_col - ly - 1:mid_col + ly] = kernel
    b_circ = torch.roll(b, (-mid_row + 1, -mid_col + 1), dims=(0, 1))

    return b_circ



def fac(y):

    temp = y.transpose(2,3).flatten(2)
    p_tem, eig_val, _ = torch.linalg.svd(torch.matmul(temp, temp.transpose(1, 2))/temp.shape[2])

    return p_tem, eig_val


def id_hs_sub(x_dh):

    p_vec, eig_val = fac(x_dh)
    nb_sub = 10
    p_vec = p_vec[:, :, :nb_sub]

    d = torch.diag_embed(torch.sqrt(eig_val[:, :nb_sub]).view(p_vec.shape[0], -1))
    p_dec = mldivide(p_vec.transpose(1, 2), d)
    p_inc = torch.matmul(p_vec, d)

    return p_inc, p_dec, d, nb_sub


if __name__ == '__main__':
    import scipy.io as io
    import numpy as np
    from Utils.spectral_tools import fspecial_gauss
    import matplotlib
    matplotlib.use('Qt5Agg')

    from matplotlib import pyplot as plt

    # temp = io.loadmat('/home/matteo/Desktop/hyperspectral_toolbox/Methods/bayesfusion/data.mat')
    temp = io.loadmat('/home/matteo/Desktop/20200124105146_002.mat')
    #ref = torch.from_numpy(np.moveaxis(io.loadmat('/home/matteo/Desktop/hyperspectral_toolbox/Demo/ref.mat')['I_REF'].astype(np.float32), -1, 0)[None, :, :, :]).double()

    pan = torch.from_numpy(temp['I_PAN'].astype(np.float64)).unsqueeze(0).unsqueeze(0)
    ms = torch.from_numpy(np.moveaxis(temp['I_MS_LR'].astype(np.float64), -1, 0)).unsqueeze(0)
    ref = torch.from_numpy(np.moveaxis(temp['I_GT'].astype(np.float64), -1, 0)).unsqueeze(0)
    pan_1 = torch.mean(ref[:, :32, :, :], dim=1, keepdim=True)
    overlap = torch.arange(0, 32).float()
    ratio = 6
    # kernel = torch.from_numpy(fspecial_gauss((9,9), (1/(2*(2.7725887)/ratio**2))**0.5))
    # kernel = torch.from_numpy(fspecial_gauss((9, 9), 0.3))
    kernel = torch.from_numpy(gen_mtf(ratio, 'PRISMA', 15, 1).squeeze()).type(ms.dtype).to(ms.device)
    fused = BayesianMethod(ms, pan, overlap, ratio, kernel, 'Gaussian')

    from Metrics.metrics_rr import ERGAS, SAM

    erg = ERGAS(ratio)(fused, ref)
    sam = SAM()(fused, ref)

    plt.figure()
    plt.imshow(ms[0,40,:,:].numpy(), cmap='gray', clim = (ref[0,0,:,:].min(), ref[0,0,:,:].max()))

    plt.figure()
    plt.imshow(ref[0,40,:,:].numpy(), cmap='gray', clim = (ref[0,0,:,:].min(), ref[0,0,:,:].max()))

    plt.figure()
    plt.imshow(fused[0,40,:,:].numpy(), cmap='gray', clim = (ref[0,0,:,:].min(), ref[0,0,:,:].max()))
    plt.show()