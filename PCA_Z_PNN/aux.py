import torch
import numpy as np
from torch import nn
from Utils.spectral_tools import gen_mtf
from math import floor
from Metrics.cross_correlation import xcorr_torch
from skimage import transform
from scipy import linalg


def normalize(img, nbits, nbands):
    return img / (np.sqrt(nbands)*(2**nbits))

def denormalize(img, nbits, nbands):
    return img * (np.sqrt(nbands)*(2**nbits))

def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False
    depthconv.to(device)
    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask


def pca(ms_lr):
    """
    Perform Principal Component Analysis (PCA) on a PyTorch tensor image.

    Args:
    ms_lr (torch.Tensor): Input image tensor of shape (1, B, H, W).

    Returns:
    pca_image (torch.Tensor): PCA-transformed image tensor with the same shape.
    pca_matrix (torch.Tensor): PCA transformation matrix.
    mean (torch.Tensor): Tensor of mean values.
    """
    # Reshape the input tensor to (B, H * W) and mean-center the data
    _, B, H, W = ms_lr.shape
    flattened = torch.reshape(ms_lr, (B, H*W))
    mean = torch.mean(flattened, dim=1).unsqueeze(1)
    centered = flattened - mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered, centered.t()) / (H * W - 1)

    # Perform PCA using SVD
    U, S, _ = torch.svd(cov_matrix)

    # PCA-transformed image
    pca_image = torch.matmul(-U.t(), centered).view(1, B, H, W)

    return pca_image, U, mean


def inverse_pca(pca_image, pca_matrix, mean):
    """
    Perform the inverse of Principal Component Analysis (PCA) on a PCA-transformed image.

    Args:
    pca_image (torch.Tensor): PCA-transformed image tensor with the same shape as the input image.
    pca_matrix (torch.Tensor): PCA transformation matrix obtained from the 'pca' function.
    mean (torch.Tensor): Tensor of mean values.

    Returns:
    original_image (torch.Tensor): Inverse PCA-reconstructed image tensor.
    """
    _, B, H, W = pca_image.shape
    flattened_pca = torch.reshape(pca_image, (B, H*W))

    flattened_image = torch.matmul(-pca_matrix, flattened_pca) + mean

    # Reconstruct the original image
    original_image = flattened_image.view(1, B, H, W)

    return original_image