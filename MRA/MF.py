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

    p_lp = p[:, :, :, :, lev - 1]

    fused = ms * (p[:, :, :, :, 0] / (p_lp + 1e-10))

    return fused


def pyr_dec(I, textse, lev, int_meth):

    p = []
    p.append(torch.unsqueeze(I, -1))

    sizes = []
    sizes.append([I.shape[2], I.shape[3]])

    first = True

    image_new = I

    for i in range(1, lev):

        image_old = image_new
        del image_new

        pd = dilation(image_old, textse)
        pe = erosion(image_old, textse)

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

        for ir in range(i, -1, -1):
            image_resized_new = resize(image_resized_old, sizes[ir], interpolation=int_meth, antialias=False)
            image_resized_old = image_resized_new
            del image_resized_new

        if torch.sum(torch.isfinite(image_resized_old)) != image_resized_old.numel():
            p = p[0].repeat(1, 1, 1, 1, lev)
            return p
        else:
            p.append(torch.unsqueeze(image_resized_old, -1))

    p = torch.concatenate(p, dim=-1)
    return p

if __name__ == '__main__':
    from scipy import io
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    from Utils.interpolator_tools import interp23tap
    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_1_zoom.mat')

    ms = interp23tap(temp['I_MS_LR'].astype(np.float32), 4)
    pan = temp['I_PAN'].astype(np.float32)

    ms = np.moveaxis(ms, -1, 0)
    ms = torch.from_numpy(ms[None, :, :, :])
    pan = torch.from_numpy(pan[None, None, :, :])

    fused = MF(ms, pan, 4)
    plt.figure()
    plt.imshow(fused[0, 0, :, :].numpy(), cmap='gray', clim=[0, 2048])
    plt.show()


    fused = fused.detach().cpu().numpy()
    fused = np.squeeze(fused)
    fused = np.moveaxis(fused, 0, -1)

    io.savemat('/home/matteo/Desktop//Adelaide_1_zoom_MF.mat', {'I_MS': fused})