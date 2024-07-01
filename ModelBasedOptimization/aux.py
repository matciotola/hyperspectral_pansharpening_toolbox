import torch
import math


def vca(R, p=0, snr_input=False, verbose='off'):
    """
    Vertex Component Analysis Algorithm [VCA]

    Args:
    R (torch.Tensor): Input matrix with dimensions L(channels) x N(pixels).
    p (int): Number of endmembers in the scene.
    snr_input (bool): Whether to use user input SNR or estimate it.
    verbose (str): Verbose mode, 'on' or 'off'.

    Returns:
    Ae (torch.Tensor): Estimated mixing matrix (endmembers signatures).
    index (torch.Tensor): Pixels chosen to be the most pure.
    Rp (torch.Tensor): Data R projected on the identified signal subspace.
    """

    # Default parameters
    if p == 0:
        p = R.shape[0]
    snr = 0

    # Initializations
    L, N = R.shape
    if (p <= 0 or p > L or p != int(p)):
        raise ValueError("ENDMEMBER parameter must be an integer between 1 and L")

    if (L - p < p) and not snr_input:
        if verbose == 'on':
            print("I can not estimate SNR [(no bands)-p < p]")
            print("I will apply the projective projection")
        snr_input = True
        snr = 100

    # Compute mean, correlation, and the p-orth basis
    r_m = torch.mean(R, dim=1)
    Corr = torch.mm(R, R.t()) / N
    Ud, Sd, _ = torch.linalg.svd(Corr)
    Ud = Ud[:, :p]

    if not snr_input:
        # Estimate SNR
        Pt = torch.trace(Corr)
        Pp = torch.sum(Sd[:p])
        Pn = (Pt - Pp) / (1 - p / L)
        if Pn > 0:
            snr = 10 * torch.log10(Pp / Pn)
            if verbose == 'on':
                print(f'SNR estimated = {snr.item():.2f} [dB]')
        else:
            snr = 0
            if verbose == 'on':
                print('Input data belongs to a p-subspace')

    # SNR threshold to decide the projection
    SNR_th = 15 + 10 * math.log10(p)

    if snr < SNR_th:
        if verbose == 'on':
            print('Select proj. on the to (p-1) subspace.')
            print('I will apply the projective projection')

        d = p - 1
        Cov = Corr - torch.outer(r_m, r_m)
        Ud, Sd, _ = torch.svd(Cov)
        Ud = Ud[:, :d]
        R_o = R - r_m.view(-1, 1)
        x_p = -Ud.t() @  R_o

        if d > x_p.shape[0]:
            Rp = torch.mm(Ud, x_p) + r_m.view(-1, 1)
        else:
            Rp = torch.mm(Ud, x_p[:d]) + r_m.view(-1, 1)

        cos_angles = torch.sum(Rp * R, dim=0) / (torch.norm(Rp, dim=0) * torch.norm(R, dim=0))
        c = torch.norm(x_p, dim=0).max()
        y = torch.cat([x_p, c * torch.ones(1, N)], dim=0)
    else:
        if verbose == 'on':
            print('... Select the projective proj. (dpft)')

        d = p
        x_p = torch.mm(Ud.t(), R)
        Rp = torch.mm(Ud, x_p[:d])
        u = torch.mean(x_p, dim=1) * p
        scale = torch.sum(x_p * u.view(-1, 1), dim=0)
        th = 0.01
        mask = scale < th
        scale = scale * (~mask) + mask.float()
        y = x_p / scale

        pt_errors = torch.nonzero(mask).squeeze()
        u_norm = torch.norm(u)
        y[:, pt_errors] = (u / (u_norm ** 2)).unsqueeze(1).repeat(1, len(pt_errors))

    # VCA algorithm
    p = y.shape[0]
    index = torch.zeros(p, dtype=torch.long)
    A = torch.zeros((p, p), dtype=R.dtype, device=R.device)
    A[-1, 0] = 1

    for i in range(p):
        w = torch.rand(p, dtype=R.dtype, device=R.device)
        f = w - A @ torch.pinverse(A) @  w
        f = f / torch.norm(f)
        v = torch.mm(f[None, :], y)
        v_max, index[i] = torch.max(torch.abs(v), dim=1)
        A[:, i] = y[:, index[i]]

    Ae = Rp[:, index]

    return Ae, index, Rp

def weights_pan_from_hs_calculation(hs, pan):
    from Utils.pansharpening_aux_tools import mldivide
    hs = hs.permute(1, 0, 2, 3).flatten(1).unsqueeze(0).transpose(1, 2)
    pan = pan.permute(1, 0, 2, 3).flatten(1).unsqueeze(0).transpose(1, 2)

    weights = torch.linalg.lstsq(pan, hs).solution.squeeze()

    weights = weights / torch.sum(weights)

    return weights




# Example usage
if __name__ == "__main__":
    from Utils.load_save_tools import open_mat
    from Utils.dl_tools import generate_paths
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    ds_root = '/media/matteo/T7/Datasets/HyperSpectral/'
    dataset = 'PRISMA'
    experiment_folder = 'Reduced_Resolution'
    ds_paths = generate_paths(ds_root, dataset, 'Test', experiment_folder)

    gt = []
    pan = []

    for path in ds_paths:
        pan_single, _, _, gt_single, _, _ = open_mat(path)
        pan.append(pan_single.float())
        gt.append(gt_single.float())

    pan = torch.cat(pan, 0)
    gt = torch.cat(gt, 0)

    weights = weights_pan_from_hs_calculation(gt, pan)

    weights = torch.round(weights, decimals=4)

    pan_tilde = weights[None, :, None, None] * gt
    pan_tilde = torch.sum(pan_tilde, dim=1, keepdim=True)

    plt.figure()
    plt.imshow(pan_tilde[0, 0, :, :].cpu().numpy())
    plt.figure()
    plt.imshow(pan[0, 0, :, :].cpu().numpy())
    plt.show()
