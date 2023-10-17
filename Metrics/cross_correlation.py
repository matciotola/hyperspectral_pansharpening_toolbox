from math import ceil
import torch
import torch.nn.functional as F


def xcorr_torch(img_1, img_2, half_width, device):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

    """
    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.double()
    img_2 = img_2.double()

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] - img_1_cum[:, :, 2 * w:, :-2 * w] +
                img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] - img_2_cum[:, :, 2 * w:, :-2 * w] +
                img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1 ** 2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2 ** 2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] - ij_cum[:, :, 2 * w:, :-2 * w] +
                   ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] - i2_cum[:, :, 2 * w:, :-2 * w] +
                   i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] - j2_cum[:, :, 2 * w:, :-2 * w] +
                   j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L
