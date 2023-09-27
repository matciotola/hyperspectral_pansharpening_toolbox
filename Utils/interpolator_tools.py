import math
import numpy as np
import scipy.ndimage.filters as ft
import torch
import torch.nn as nn
from torchvision.transforms.functional import pad
from torch.nn.functional import conv2d

def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'
    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img


def interp23tap_torch(img, ratio):
    """
        A PyTorch implementation of the Polynomial interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """
    device = img.device
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    bs, b, r, c = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))
    BaseCoeff = np.concatenate([BaseCoeff] * b, axis=0)

    BaseCoeff = torch.from_numpy(BaseCoeff).to(device)
    #img = img.astype(np.float32)
    #img = np.moveaxis(img, -1, 0)

    for z in range(int(ratio / 2)):

        I1LRU = torch.zeros((bs, b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c), device=device, dtype=BaseCoeff.dtype)

        if z == 0:
            I1LRU[:, :, 1::2, 1::2] = img
        else:
            I1LRU[:, :, ::2, ::2] = img

        #I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(I1LRU, 2, 3))
        img = conv(torch.transpose(t, 2, 3))#.cpu().detach().numpy()
        #img = np.squeeze(img)

    #img = np.moveaxis(img, 0, -1)

    return img



def interp_3x_1d(img, N=50):
    ratio = 3

    bs, c, h, w = img.shape
    n = torch.arange(-N, N + 1)
    h1 = torch.sinc(n + 1 / ratio)
    h1 = h1 / torch.sum(h1)

    h1 = torch.fliplr(h1[None, :])
    h1 = h1[None, None, :, :]

    h2 = torch.sinc(n + 2 / ratio)
    h2 = h2 / torch.sum(h2)
    h2 = torch.fliplr(h2[None, :])
    h2 = h2[None, None, :, :]

    h1 = h1.repeat(c, 1, 1, 1).type(img.dtype).to(img.device)
    h2 = h2.repeat(c, 1, 1, 1).type(img.dtype).to(img.device)


    img_padded = pad(img, [N+1, 0, N , 0], padding_mode='symmetric')

    x1 = conv2d(img_padded, h1, padding='same', groups=c)
    x1 = x1[:, :, :, N+1:-N]

    x2 = conv2d(img_padded, h2, padding='same', groups=c)
    x2 = x2[:, :, :, N:-N-1]

    y = torch.zeros((bs, c, h, w * ratio), device=img.device, dtype=img.dtype)

    y[:, :, :, ::ratio] = x2
    y[:, :, :, 1::ratio] = img
    y[:, :, :, 2::ratio] = x1

    return y


def interp_3x_2d(img, N=50):

    z = interp_3x_1d(img, N)
    z = interp_3x_1d(z.transpose(2, 3), N)
    z = z.transpose(2, 3)
    return z


def ideal_interpolator(img, ratio):

    if ratio == 2:
        img_upsampled = interp23tap_torch(img, ratio)
    else:
        img_upsampled = interp_3x_2d(img)
        img_upsampled = interp23tap_torch(img_upsampled, 2)

    return img_upsampled