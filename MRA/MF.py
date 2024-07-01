import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter
import math

from kornia.morphology import dilation, erosion

def MF(ordered_dict):

    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    # Equalization

    pan = pan.repeat(1, ms.shape[1], 1, 1)
    pan = (pan - torch.mean(pan, dim=(2, 3), keepdim=True)) * (torch.std(ms, dim=(2,3), keepdim=True) / torch.std(pan, dim=(2,3), keepdim=True)) + torch.mean(ms, dim=(2,3), keepdim=True)

    # structuring Element choice

    textse = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=pan.dtype, device=pan.device)
    int_meth = Inter.BILINEAR
    lev = math.ceil(math.log2(ratio)) + 1

    # Image construction

    p = pyr_dec(pan, textse, lev, int_meth)

    p_lp = p[:, :, :, :, - 1]

    fused = ms * torch.clip((p[:, :, :, :, 0] / (p_lp + torch.finfo(torch.float64).eps)), 0, 10.0)

    return fused


def pyr_dec(I, textse, lev, int_meth):
    p = []
    p.append(torch.unsqueeze(I, -1))

    sizes = []
    sizes.append([I.shape[2], I.shape[3]])

    first = True

    image_new = I

    for i in range(1, lev):

        image_old = torch.clone(image_new)
        del image_new

        pd = dilation(image_old, textse, max_val=torch.inf)

        #pe = []
        # for kk in range(image_old.shape[1]):
        pe = erosion(image_old, textse, max_val=torch.inf)
        # pe = torch.cat(pe, dim=1)

        rho_minus = image_old - pe
        rho_plus = pd - image_old

        d = rho_minus - rho_plus
        ps = image_old - 0.5 * d

        # downsampling

        if first:
            image_new = ps[:, :, 1::2, 1::2]
            first = False
        else:
            image_new = ps[:, :, ::2, ::2]

        sizes.append([image_new.shape[2], image_new.shape[3]])

        image_resized_old = image_new

        for ir in range(i-1, -1, -1):
            image_resized_new = resize(image_resized_old, sizes[ir], interpolation=int_meth, antialias=True)
            image_resized_old = image_resized_new
            del image_resized_new

        if torch.sum(torch.isfinite(image_resized_old)) != image_resized_old.numel():
            p = p[0].repeat(1, 1, 1, 1, lev)
            return p
        else:
            p.append(torch.unsqueeze(image_resized_old, -1))

    p = torch.concatenate(p, dim=-1)
    return p
