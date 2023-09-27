"""
BM3D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise.

based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189

Copyright (c) 2006-2019 Tampere University.
All rights reserved.
This work (software, material, and documentation) shall only
be used for nonprofit noncommercial purposes.
Any unauthorized use of this work for commercial or for-profit purposes
is prohibited.
"""


import numpy as np
import pywt
import copy
import os
from typing import Union, Tuple
from scipy.fftpack import *
from scipy.linalg import *
from scipy.ndimage.filters import correlate
from scipy.io import loadmat
from scipy import signal
from scipy.interpolate import interpn

# PyWavelets is not essential, as we include a few hard-coded transforms
try:
    import pywt
except ImportError:
    pywt = None

from .bm3d_ctypes import bm3d_step
# Alternatively, from .bm3d_c import bm3d_step

from .profiles import BM3DProfile, BM3DProfileRefilter, BM3DProfileVN, BM3DStages
from .profiles import BM3DProfileDeb, BM3DProfileHigh, BM3DProfileLC, BM3DProfileVNOld

EPS = 2.2204e-16

def bm3d_rgb_mod(z: np.ndarray, sigma_psd: Union[np.ndarray, list, float],
             profile: Union[BM3DProfile, str] = 'np', colorspace: str = 'opp')\
        -> np.ndarray:
    """
    BM3D For color images. Performs color transform to do block-matching in luminance domain.
    :param z: Noisy image, 3 channels (MxNx3)
    :param sigma_psd: Noise PSD, either MxN or MxNx3 (different PSDs for different channels)
                        or
                      Noise standard deviation, either float, or [float, float, float] for 3 different stds.
    :param profile: Settings for BM3D: BM3DProfile object or a string.
                    ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
    :param colorspace: 'YCbCr' or 'opp' for choosing the color transform
    :return: denoised color image, same size as z
    """

    # Forward color transform
    sigma = np.sqrt(np.sum((np.asarray(sigma_psd) / len(sigma_psd)) ** 2))
    #sigma_psd = list(np.asarray(sigma_psd) / sigma / 48)
    #_, imax, imin, scale, a = rgb_to(z, colorspace)

    # Scale PSD appropriately

    # Multiple PSDs / sigmas
    """
    if np.ndim(sigma_psd) == 1 or (np.ndim(sigma_psd) == 3 and sigma_psd.shape[2] == 3):
        sigma_psd = np.array(sigma_psd)
        if np.ndim(sigma_psd) == 3:
            o = sigma_psd.reshape([sigma_psd.shape[0] * sigma_psd.shape[1], 3]) @ (a.T ** 2)
            o = o.reshape([sigma_psd.shape[0], sigma_psd.shape[1], 3])
            sigma_psd = o / np.transpose(np.atleast_3d((imax - imin) ** 2), (0, 2, 1))
        else:
            # One-dim PSDs
            o = np.array(np.ravel(sigma_psd)).T ** 2 @ (a.T ** 2)
            sigma_psd = np.sqrt(o / (imax - imin) ** 2)
    else:
        if np.squeeze(sigma_psd).ndim <= 1:  # stds are scaled by the sqrt.
            sigma_psd = sigma_psd * np.transpose(np.atleast_3d(np.sqrt(scale)), (0, 2, 1))
        else:
            sigma_psd = np.atleast_3d(sigma_psd) * np.transpose(np.atleast_3d(scale), (0, 2, 1))
    """
    sigma_psd = sigma_psd / sigma / 255.0
    # Call BM3D with the transformed image and PSD
    y_hat = bm3d(z, sigma_psd, profile)

    # Inverse transform to get the final estimate
    #y_hat, imax, imin, scale, a = rgb_to(y_hat, colorspace, True, imax, imin)

    return y_hat



def bm3d_rgb(z: np.ndarray, sigma_psd: Union[np.ndarray, list, float],
             profile: Union[BM3DProfile, str] = 'np', colorspace: str = 'opp')\
        -> np.ndarray:
    """
    BM3D For color images. Performs color transform to do block-matching in luminance domain.
    :param z: Noisy image, 3 channels (MxNx3)
    :param sigma_psd: Noise PSD, either MxN or MxNx3 (different PSDs for different channels)
                        or
                      Noise standard deviation, either float, or [float, float, float] for 3 different stds.
    :param profile: Settings for BM3D: BM3DProfile object or a string.
                    ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
    :param colorspace: 'YCbCr' or 'opp' for choosing the color transform
    :return: denoised color image, same size as z
    """

    # Forward color transform
    z, imax, imin, scale, a = rgb_to(z, colorspace)

    # Scale PSD appropriately

    # Multiple PSDs / sigmas

    if np.ndim(sigma_psd) == 1 or (np.ndim(sigma_psd) == 3 and sigma_psd.shape[2] == 3):
        sigma_psd = np.array(sigma_psd)
        if np.ndim(sigma_psd) == 3:
            o = sigma_psd.reshape([sigma_psd.shape[0] * sigma_psd.shape[1], 3]) @ (a.T ** 2)
            o = o.reshape([sigma_psd.shape[0], sigma_psd.shape[1], 3])
            sigma_psd = o / np.transpose(np.atleast_3d((imax - imin) ** 2), (0, 2, 1))
        else:
            # One-dim PSDs
            o = np.array(np.ravel(sigma_psd)).T ** 2 @ (a.T ** 2)
            sigma_psd = np.sqrt(o / (imax - imin) ** 2)
    else:
        if np.squeeze(sigma_psd).ndim <= 1:  # stds are scaled by the sqrt.
            sigma_psd = sigma_psd * np.transpose(np.atleast_3d(np.sqrt(scale)), (0, 2, 1))
        else:
            sigma_psd = np.atleast_3d(sigma_psd) * np.transpose(np.atleast_3d(scale), (0, 2, 1))

    # Call BM3D with the transformed image and PSD
    y_hat = bm3d(z, sigma_psd, profile)

    # Inverse transform to get the final estimate
    y_hat, imax, imin, scale, a = rgb_to(y_hat, colorspace, True, imax, imin)

    return y_hat


def bm3d_deblurring(z: np.ndarray,
                    sigma_psd: Union[np.ndarray, list, float],
                    psf: np.ndarray, profile: Union[BM3DProfile, str] = 'np')\
        -> np.ndarray:
    """
    BM3D Deblurring. Performs regularization, then denoising.
    :param z: Noisy blurred image. either MxN or MxNxC where C is the channel count.
    :param sigma_psd: Noise PSD, either MxN or MxNxC (different PSDs for different channels)
                        or
                      sigma_psd: Noise standard deviation, either float, or length C list of floats
    :param psf: Blur point-spread function in space domain.
    :param profile: Settings for BM3D: BM3DProfile object or a string.
                    ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
    :return: denoised, deblurred image, same size as z
    """

    # Handle both PSDs and sigmas
    sigma_psd = np.array(sigma_psd)

    # Create big PSD
    if np.squeeze(sigma_psd).ndim <= 1:
        sigma_psd = np.ones(z.shape) * np.ravel(sigma_psd ** 2).reshape([1, 1, np.size(sigma_psd)]) * \
                    z.shape[0] * z.shape[1]

    sigma_psd = np.atleast_3d(sigma_psd)
    z = np.atleast_3d(z)

    # Regularized inverse
    regularization_alpha_ri = 4e-4

    # pad PSF with zeros to whole image domain, and center it
    big_v = np.zeros(z.shape[0:2])
    big_v[0:psf.shape[0], 0:psf.shape[1]] = psf
    big_v = np.roll(big_v, -np.array(np.round([(psf.shape[0] - 1) / 2,
                                               (psf.shape[1] - 1) / 2]), dtype=int), axis=(0, 1))

    # PSF in FFT
    fft_v = np.atleast_3d(fft2(big_v, axes=(0, 1)))

    # Standard Tikhonov Regularization
    regularized_inverse = np.conj(fft_v) / ((np.abs(fft_v) ** 2) + regularization_alpha_ri * sigma_psd + EPS)

    # Regularized Inverse Estimate (RI OBSERVATION)
    z_ri = np.real(ifft2(fft2(z, axes=(0, 1)) * regularized_inverse, axes=(0, 1)))

    # PSD of estimate
    sigma_psd_ri = sigma_psd * abs(regularized_inverse) ** 2

    # Call BM3D hard-thresholding with the RI and its PSD
    y_hat = bm3d(z_ri, sigma_psd_ri, profile, stage_arg=BM3DStages.HARD_THRESHOLDING)

    # Regularized Wiener Inversion
    regularization_alpha_rwi = 5e-3

    # Wiener reference estimate
    wiener_pilot = np.atleast_3d(abs(fft2(y_hat, axes=(0, 1))))

    # Transfer Matrix for RWI(uses standard regularization 'a-la-Tikhonov')
    regularized_wiener_inverse = (np.conj(fft_v) * wiener_pilot ** 2) / (wiener_pilot ** 2 * (np.abs(fft_v) ** 2) +
                                                                         regularization_alpha_rwi * sigma_psd + EPS)

    # Regularized Wiener inverse
    z_rwi = np.real(ifft2(fft2(z, axes=(0, 1)) * regularized_wiener_inverse, axes=(0, 1)))
    # And its PSD
    sigma_psd_rwi = sigma_psd * np.abs(regularized_wiener_inverse) ** 2

    # Filter zRWI in Wiener using the HT result as pilot
    return bm3d(z_rwi, sigma_psd_rwi, profile, stage_arg=y_hat)


def bm3d(z: np.ndarray, sigma_psd: Union[np.ndarray, list, float],
         profile: Union[BM3DProfile, str] = 'np',
         stage_arg: Union[BM3DStages, np.ndarray] = BM3DStages.ALL_STAGES,
         blockmatches: tuple = (False, False))\
        -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Perform BM3D denoising on z: either hard-thresholding, Wiener filtering or both.

    :param z: Noisy image. either MxN or MxNxC where C is the channel count.
              For multichannel images, blockmatching is performed on the first channel.
    :param sigma_psd: Noise PSD, either MxN or MxNxC (different PSDs for different channels)
            or
           sigma_psd: Noise standard deviation, either float, or length C list of floats
    :param profile: Settings for BM3D: BM3DProfile object or a string
                    ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb'). Default 'np'.
    :param stage_arg: Determines whether to perform hard-thresholding or wiener filtering.
                    either BM3DStages.HARD_THRESHOLDING, BM3DStages.ALL_STAGES or an estimate
                    of the noise-free image.
                    - BM3DStages.ALL_STAGES: Perform both.
                    - BM3DStages.HARD_THRESHOLDING: Perform hard-thresholding only.
                    - ndarray, size of z: Perform Wiener Filtering with stage_arg as pilot.
    :param blockmatches: Tuple (HT, Wiener), with either value either:
                        - False : Do not save blockmatches for phase
                        - True : Save blockmatches for phase
                        - Pre-computed block-matching array returned by a previous call with [True]
                        Such as y_est, matches = BM3D(z, sigma_psd, profile, blockMatches=(True, True))
                        y_est2 = BM3D(z2, sigma_psd, profile, blockMatches=matches);
    :return:
        - denoised image, same size as z: if blockmatches == (False, False)
        - denoised image, blockmatch data: if either element of blockmatches is True
    """

    # Profile selection, if profile is a string, otherwise BM3DProfile.
    pro = _select_profile(profile)

    # Ensure z is 3-D a numpy array
    z = np.array(z)
    if z.ndim == 1:
        raise ValueError("z must be either a 2D or a 3D image!")
    if z.ndim == 2:
        z = np.atleast_3d(z)

    # If profile defines maximum required pad, use that, otherwise use image size
    pad_size = (int(np.ceil(z.shape[0] / 2)), int(np.ceil(z.shape[1] / 2)))\
        if pro.max_pad_size is None else pro.max_pad_size

    # Add 3rd dimension if necessary - don't pad there
    if z.ndim == 3 and len(pad_size) == 2:
        pad_size = (pad_size[0], pad_size[1], 0)

    y_hat = None
    ht_blocks = None
    wie_blocks = None

    # If we passed a numpy array as stage_arg, presume it is a hard-thresholding estimate.
    if isinstance(stage_arg, np.ndarray):
        y_hat = np.atleast_3d(stage_arg)
        stage_arg = BM3DStages.WIENER_FILTERING
        if y_hat.shape != z.shape:
            raise ValueError("Estimate passed in stage_arg must be equal size to z!")

    elif stage_arg == BM3DStages.WIENER_FILTERING:
        raise ValueError("If you wish to only perform Wiener filtering, you need to pass an estimate as stage_arg!")

    # If this is true, we are doing hard thresholding (whether we do Wiener later or not)
    stage_ht = (stage_arg.value & BM3DStages.HARD_THRESHOLDING.value) != 0
    # If this is true, we are doing Wiener filtering
    stage_wie = (stage_arg.value & BM3DStages.WIENER_FILTERING.value) != 0

    channel_count = z.shape[2]
    sigma_psd = np.array(sigma_psd)
    single_dim_psd = False

    # Format single dimension (std) sigma_psds
    if np.squeeze(sigma_psd).ndim <= 1:
        single_dim_psd = True
    if np.squeeze(sigma_psd).ndim == 1:
        sigma_psd = np.atleast_3d(np.ravel(sigma_psd)).transpose(0, 2, 1)
    else:
        sigma_psd = np.atleast_3d(sigma_psd)

    # Handle blockmatching inputs
    blockmatches_ht, blockmatches_wie = blockmatches  # Break apart

    # Convert blockmatch args to array even if they're single value
    if type(blockmatches_ht) == bool:
        blockmatches_ht = np.array([blockmatches_ht], dtype=np.int32)
    if type(blockmatches_wie) == bool:
        blockmatches_wie = np.array([blockmatches_wie], dtype=np.int32)

    sigma_psd2, psd_blur, psd_k = _process_psd(sigma_psd, z, single_dim_psd, pad_size, pro)

    # Step 1. Produce the basic estimate by HT filtering
    if stage_ht:

        # Get used transforms and aggregation windows.
        t_forward, t_inverse, hadper_trans_single_den,\
            inverse_hadper_trans_single_den, wwin2d = _get_transforms(pro, True)

        # Call the actual hard-thresholding step with the acquired parameters
        y_hat, ht_blocks = bm3d_step(BM3DStages.HARD_THRESHOLDING, z, psd_blur, single_dim_psd,
                                     pro, t_forward, t_inverse.T, hadper_trans_single_den,
                                     inverse_hadper_trans_single_den, wwin2d, channel_count, blockmatches_ht)
        if pro.print_info:
            print('Hard-thresholding stage completed')

        # Residual denoising, HT
        if pro.denoise_residual:

                remains, remains_psd = get_filtered_residual(z, y_hat, sigma_psd2, pad_size, pro.residual_thr)
                remains_psd = _process_psd_for_nf(remains_psd, psd_k, pro)

                if np.min(np.max(np.max(remains_psd, axis=0), axis=0)) > 1e-5:
                    # Re-filter
                    y_hat, ht_blocks = bm3d_step(BM3DStages.HARD_THRESHOLDING, y_hat + remains, remains_psd, False,
                                                 pro, t_forward, t_inverse.T, hadper_trans_single_den,
                                                 inverse_hadper_trans_single_den, wwin2d, channel_count,
                                                 blockmatches_ht, refiltering=True)

    # Step 2. Produce the final estimate by Wiener filtering (using the
    # hard-thresholding initial estimate)
    if stage_wie:

        # Get used transforms and aggregation windows.
        t_forward, t_inverse, hadper_trans_single_den,\
            inverse_hadper_trans_single_den, wwin2d = _get_transforms(pro, False)

        # Multiply PSDs by mus
        mu_list = np.ravel(pro.lambda_wiener).reshape([1, 1, np.size(pro.lambda_wiener)])
        if single_dim_psd:
            mu_list = np.sqrt(mu_list)
        psd_blur_mult = psd_blur * mu_list

        # Wiener filtering
        y_hat, wie_blocks = bm3d_step(BM3DStages.WIENER_FILTERING, z, psd_blur_mult, single_dim_psd,
                                      pro, t_forward, t_inverse.T, hadper_trans_single_den,
                                      inverse_hadper_trans_single_den, wwin2d, channel_count,
                                      blockmatches_wie, y_hat=y_hat)

        # Residual denoising, Wiener
        if pro.denoise_residual:
            remains, remains_psd = get_filtered_residual(z, y_hat, sigma_psd2, pad_size, pro.residual_thr)
            remains_psd = _process_psd_for_nf(remains_psd, psd_k, pro)

            if np.min(np.max(np.max(remains_psd, axis=0), axis=0)) > 1e-5:

                psd_blur_mult = remains_psd * np.ravel(pro.lambda_wiener2).reshape([1, 1, np.size(pro.lambda_wiener2)])
                y_hat, wie_blocks = bm3d_step(BM3DStages.WIENER_FILTERING, y_hat + remains, psd_blur_mult, False,
                                              pro, t_forward, t_inverse.T, hadper_trans_single_den,
                                              inverse_hadper_trans_single_den, wwin2d, channel_count, blockmatches_wie,
                                              refiltering=True, y_hat=y_hat)

        if pro.print_info:
            print('Wiener-filtering stage completed')

    if not stage_ht and not stage_wie:
        raise ValueError("No operation was selected!")

    # Remove useless dimension if only single output
    if channel_count == 1:
        y_hat = y_hat[:, :, 0]

    if blockmatches_ht[0] == 1 and blockmatches_wie[0] != 1:  # We computed & want to return block-matches for HT
        return y_hat, (ht_blocks, np.zeros(1, dtype=np.intc))
    if blockmatches_ht[0] == 1:  # Both
        return y_hat, (ht_blocks, wie_blocks)
    if blockmatches_ht[0] != 1 and blockmatches_wie[0] == 1:  # Only Wiener
        return y_hat, (np.zeros(1, dtype=np.intc), wie_blocks)

    return y_hat


def get_filtered_residual(z: np.ndarray, y_hat: np.ndarray, sigma_psd: Union[np.ndarray, float],
                          pad_size: Union[list, tuple], residual_thr: float) -> (np.ndarray, np.ndarray):
    """
    Get residual, filtered by global FFT HT
    :param z: Original noisy image (MxNxC)
    :param y_hat: Estimate of noise-free image, same size as z
    :param sigma_psd: std, 1-D list of stds or MxNx1 or MxNxC "list" of PSDs.
            Note! if PSD, the size must be size of z + 2 * pad_size, not size of z!
    :param pad_size: amount to pad around z and y_hat to avoid problems due to non-circular noise.
                     Should be at least kernel size in total (1/2 on one side), but may be bigger if kernel size
                     is unknown.
    :param residual_thr: The threshold to use in the global Fourier filter.
    :return: (filtered residual, same size as z, PSD of the filtered residual, same size as z)

    """

    # Calculate the residual
    if pad_size[0]:
        pads_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0))
        resid = fft2(np.pad(z - y_hat, pads_width, 'constant'), axes=(0, 1))
    else:
        resid = fft2(z - y_hat, axes=(0, 1))

    # Kernel size for dilation
    ksz = [np.ceil(resid.shape[0] / 150), np.ceil(resid.shape[1] / 150)]
    ksz = [ksz[0] + 1 - (ksz[0] % 2), ksz[1] + 1 - (ksz[1] % 2)]

    psd_size_div = (z.shape[0] * z.shape[1])
    psd = sigma_psd
    if sigma_psd.shape[2] == sigma_psd.size:
        psd = sigma_psd * sigma_psd * psd_size_div

    # Apply dilation filter
    kernel = np.atleast_3d(gaussian_kernel(ksz, resid.shape[0] / 500, resid.shape[1] / 500))
    cc = correlate(np.array(np.abs(resid) > (residual_thr * np.sqrt(psd)), dtype=np.float), kernel, mode='wrap')

    # Threshold mask
    msk = (cc > 0.01)

    # Residual + PSD
    remains = np.real(ifft2(resid * msk, axes=(0, 1)))
    remains_psd = psd * msk

    # Cut off padding
    remains = remains[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]

    temp_kernel = np.real(fftshift(ifft2(np.sqrt(remains_psd / psd_size_div), axes=(0, 1)), axes=(0, 1)))
    temp_kernel = temp_kernel[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]

    remains_psd = np.power(abs(fft2(temp_kernel, z.shape[:-1], axes=(0, 1))), 2) * (z.shape[0] * z.shape[1])

    return remains, remains_psd


def gaussian_kernel(size: tuple, std: float, std2: float = -1) -> np.ndarray:
    """
    Get a 2D Gaussian kernel of size (sz1, sz2) with the specified standard deviations.
    If std2 is not specified, both stds will be the same.
    :param size: kernel size, tuple
    :param std: std of 1st dimension
    :param std2: std of 2nd dimension, or -1 if equal to std
    :return: normalized Gaussian kernel (sum == 1)
    """
    if std2 == -1:
        std2 = std
    g1d = signal.gaussian(int(size[0]), std=std).reshape(int(size[0]), 1)
    g1d2 = signal.gaussian(int(size[1]), std=std2).reshape(int(size[1]), 1)

    g2d = np.outer(g1d / np.sum(g1d), g1d2 / np.sum(g1d2))
    return g2d


def _process_psd_for_nf(sigma_psd: np.ndarray, psd_k: Union[np.ndarray, None], profile: BM3DProfile)\
        -> np.ndarray:
    """
    Process PSD so that Nf-size PSD is usable.
    :param sigma_psd: the PSD
    :param psd_k: a previously generated kernel to convolve the PSD with, or None if not used
    :param profile: the profile used
    :return: processed PSD
    """
    if profile.nf == 0:
        return sigma_psd

    # Reduce PSD size to start with
    max_ratio = 16
    sigma_psd_copy = np.copy(sigma_psd)
    single_kernel = np.ones((3, 3, 1)) / 9
    orig_ratio = np.max(sigma_psd.shape) / profile.nf
    ratio = orig_ratio
    while ratio > max_ratio:
        mid_corr = correlate(sigma_psd_copy, single_kernel, mode='wrap')
        sigma_psd_copy = mid_corr[1::3, 1::3]
        ratio = np.max(sigma_psd_copy.shape) / profile.nf

    # Scale PSD because the binary expects it to be scaled by size
    sigma_psd_copy *= (ratio / orig_ratio) ** 2
    if psd_k is not None:
        sigma_psd_copy = correlate(sigma_psd_copy, psd_k, mode='wrap')

    return sigma_psd_copy


def _select_profile(profile: Union[str, BM3DProfile]) -> BM3DProfile:
    """
    Select profile for BM3D
    :param profile: BM3DProfile or a string
    :return: BM3DProfile object
    """
    if isinstance(profile, BM3DProfile):
        pro = copy.copy(profile)
    elif profile == 'np':
        pro = BM3DProfile()
    elif profile == 'refilter':
        pro = BM3DProfileRefilter()
    elif profile == 'vn':
        pro = BM3DProfileVN()
    elif profile == 'high':
        pro = BM3DProfileHigh()
    elif profile == 'vn_old':
        pro = BM3DProfileVNOld()
    elif profile == 'deb':
        pro = BM3DProfileDeb()
    else:
        raise TypeError('"profile" should be either a string of '
                        '"np"/"refilter"/"vn"/"high"/"vn_old"/"deb" or a BM3DProfile object!')
    return pro


def _get_transf_matrix(n: int, transform_type: str,
                       dec_levels: int = 0, flip_hardcoded: bool = False) -> (np.ndarray, np.ndarray):
    """
    Create forward and inverse transform matrices, which allow for perfect
    reconstruction. The forward transform matrix is normalized so that the
    l2-norm of each basis element is 1.
    Includes hardcoded transform matrices which are kept for matlab compatibility

    :param n: Transform size (nxn)
    :param transform_type: Transform type 'dct', 'dst', 'hadamard', or anything that is
                           supported by 'wavedec'
                           'DCrand' -- an orthonormal transform with a DC and all
                           the other basis elements of random nature
    :param dec_levels:  If a wavelet transform is generated, this is the
                           desired decomposition level. Must be in the
                           range [0, log2(N)-1], where "0" implies
                           full decomposition.
    :param flip_hardcoded: Return transpose of the hardcoded matrices.
    :return: (forward transform, inverse transform)
    """

    if n == 1:
        t_forward = 1
    elif transform_type == 'hadamard':
        t_forward = hadamard(n)
    elif n == 8 and transform_type == 'bior1.5':
        t_forward = [[0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110,
                      0.343550200747110, 0.343550200747110, 0.343550200747110],
                     [-0.225454819240296, -0.461645582253923, -0.461645582253923, -0.225454819240296, 0.225454819240296,
                      0.461645582253923, 0.461645582253923, 0.225454819240296],
                     [0.569359398342840, 0.402347308162280, -0.402347308162280, -0.569359398342840, -0.083506045090280,
                      0.083506045090280, -0.083506045090280, 0.083506045090280],
                     [-0.083506045090280, 0.083506045090280, -0.083506045090280, 0.083506045090280, 0.569359398342840,
                      0.402347308162280, -0.402347308162280, -0.569359398342840],
                     [0.707106781186550, -0.707106781186550, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0.707106781186550, -0.707106781186550, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0.707106781186550, -0.707106781186550, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0.707106781186550, -0.707106781186550]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dct':
        t_forward = [[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
                      0.353553390593274, 0.353553390593274, 0.353553390593274],
                     [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064, -0.097545161008064,
                      -0.277785116509801, -0.415734806151273, -0.490392640201615],
                     [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643,
                      -0.191341716182545, 0.191341716182545, 0.461939766255643],
                     [0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801, 0.277785116509801,
                      0.490392640201615, 0.097545161008064, -0.415734806151273],
                     [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274,
                      -0.353553390593274, -0.353553390593274, 0.353553390593274],
                     [0.277785116509801, -0.490392640201615, 0.097545161008064, 0.415734806151273, -0.415734806151273,
                      -0.097545161008064, 0.490392640201615, -0.277785116509801],
                     [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545,
                      0.461939766255643, -0.461939766255643, 0.191341716182545],
                     [0.097545161008064, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615,
                      -0.415734806151273, 0.277785116509801, -0.097545161008064]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 11 and transform_type == 'dct':
        t_forward = [[0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764,
                      0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764,
                      0.301511344577764],
                     [0.422061280946316, 0.387868386059133, 0.322252701275551, 0.230530019145232, 0.120131165878581,
                      -8.91292406723889e-18, -0.120131165878581, -0.230530019145232, -0.322252701275551,
                      -0.387868386059133, -0.422061280946316],
                     [0.409129178625571, 0.279233555180591, 0.0606832509357945, -0.177133556713755, -0.358711711672592,
                      -0.426401432711221, -0.358711711672592, -0.177133556713755, 0.0606832509357945, 0.279233555180591,
                      0.409129178625571],
                     [0.387868386059133, 0.120131165878581, -0.230530019145232, -0.422061280946316, -0.322252701275551,
                      1.71076608154014e-17, 0.322252701275551, 0.422061280946316, 0.230530019145232, -0.120131165878581,
                      -0.387868386059133],
                     [0.358711711672592, -0.0606832509357945, -0.409129178625571, -0.279233555180591, 0.177133556713755,
                      0.426401432711221, 0.177133556713755, -0.279233555180591, -0.409129178625571, -0.0606832509357945,
                      0.358711711672592],
                     [0.322252701275551, -0.230530019145232, -0.387868386059133, 0.120131165878581, 0.422061280946316,
                      -8.13580150049806e-17, -0.422061280946316, -0.120131165878581, 0.387868386059133,
                      0.230530019145232, -0.322252701275551],
                     [0.279233555180591, -0.358711711672592, -0.177133556713755, 0.409129178625571, 0.0606832509357945,
                      -0.426401432711221, 0.0606832509357944, 0.409129178625571, -0.177133556713755, -0.358711711672592,
                      0.279233555180591],
                     [0.230530019145232, -0.422061280946316, 0.120131165878581, 0.322252701275551, -0.387868386059133,
                      -2.87274927630557e-18, 0.387868386059133, -0.322252701275551, -0.120131165878581,
                      0.422061280946316, -0.230530019145232],
                     [0.177133556713755, -0.409129178625571, 0.358711711672592, -0.0606832509357945, -0.279233555180591,
                      0.426401432711221, -0.279233555180591, -0.0606832509357944, 0.358711711672592, -0.409129178625571,
                      0.177133556713755],
                     [0.120131165878581, -0.322252701275551, 0.422061280946316, -0.387868386059133, 0.230530019145232,
                      2.03395037512452e-17, -0.230530019145232, 0.387868386059133, -0.422061280946316,
                      0.322252701275551,
                      -0.120131165878581],
                     [0.0606832509357945, -0.177133556713755, 0.279233555180591, -0.358711711672592, 0.409129178625571,
                      -0.426401432711221, 0.409129178625571, -0.358711711672592, 0.279233555180591, -0.177133556713755,
                      0.0606832509357945]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dst':
        t_forward = [[0.161229841765317, 0.303012985114696, 0.408248290463863, 0.464242826880013, 0.464242826880013,
                      0.408248290463863, 0.303012985114696, 0.161229841765317],
                     [0.303012985114696, 0.464242826880013, 0.408248290463863, 0.161229841765317, -0.161229841765317,
                      -0.408248290463863, -0.464242826880013, -0.303012985114696],
                     [0.408248290463863, 0.408248290463863, 0, -0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, 0.408248290463863],
                     [0.464242826880013, 0.161229841765317, -0.408248290463863, -0.303012985114696, 0.303012985114696,
                      0.408248290463863, -0.161229841765317, -0.464242826880013],
                     [0.464242826880013, -0.161229841765317, -0.408248290463863, 0.303012985114696, 0.303012985114696,
                      -0.408248290463863, -0.161229841765317, 0.464242826880013],
                     [0.408248290463863, -0.408248290463863, 0, 0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, -0.408248290463863],
                     [0.303012985114696, -0.464242826880013, 0.408248290463863, -0.161229841765317, -0.161229841765317,
                      0.408248290463863, -0.464242826880013, 0.303012985114696],
                     [0.161229841765317, -0.303012985114696, 0.408248290463863, -0.464242826880013, 0.464242826880013,
                      -0.408248290463863, 0.303012985114696, -0.161229841765317]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif transform_type == 'dct':
        t_forward = dct(np.eye(n), norm='ortho')
    elif transform_type == 'eye':
        t_forward = np.eye(n)
    elif transform_type == 'dst':
        t_forward = dst(np.eye(n), norm='ortho')
    elif transform_type == 'DCrand':
        x = np.random.normal(n)
        x[:, 0] = np.ones(len(x[:, 0]))
        q, _, _ = np.linalg.qr(x)
        if q[0] < 0:
            q = -q

        t_forward = q.T

    elif pywt is not None:
        # a wavelet decomposition supported by PyWavelets
        # Set periodic boundary conditions, to preserve bi-orthogonality
        t_forward = np.zeros((n, n))

        for ii in range(n):
            temp = np.zeros(n)
            temp[0] = 1.0
            temp = np.roll(temp, (ii, dec_levels))
            tt = pywt.wavedec(temp, transform_type, mode='periodization', level=int(np.log2(n)))
            cc = np.hstack(tt)
            t_forward[:, ii] = cc

    else:
        raise ValueError("Transform of " + transform_type + "couldn't be found and PyWavelets couldn't be imported!")

    t_forward = np.array(t_forward)
    # Normalize the basis elements
    if not ((n == 8) and transform_type == 'bior1.5'):
        try:
            t_forward = (t_forward.T @ np.diag(np.sqrt(1. / sum(t_forward ** 2, 0)))).T
        except TypeError:  # t_forward was not an array...
            pass

    # Compute the inverse transform matrix
    try:
        t_inverse = np.linalg.inv(t_forward)
    except LinAlgError:
        t_inverse = np.array([[1]])

    return t_forward, t_inverse

def _get_transf_matrix_mod(n: int, transform_type: str,
                       dec_levels: int = 0, flip_hardcoded: bool = False) -> (np.ndarray, np.ndarray):
    """
    Create forward and inverse transform matrices, which allow for perfect
    reconstruction. The forward transform matrix is normalized so that the
    l2-norm of each basis element is 1.
    Includes hardcoded transform matrices which are kept for matlab compatibility

    :param n: Transform size (nxn)
    :param transform_type: Transform type 'dct', 'dst', 'hadamard', or anything that is
                           supported by 'wavedec'
                           'DCrand' -- an orthonormal transform with a DC and all
                           the other basis elements of random nature
    :param dec_levels:  If a wavelet transform is generated, this is the
                           desired decomposition level. Must be in the
                           range [0, log2(N)-1], where "0" implies
                           full decomposition.
    :param flip_hardcoded: Return transpose of the hardcoded matrices.
    :return: (forward transform, inverse transform)
    """

    if n == 1:
        t_forward = 1
    elif transform_type == 'hadamard':
        t_forward = hadamard(n)
    elif n == 8 and transform_type == 'bior1.5':
        t_forward = [
                    [0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,],
                    [0.219417649252501,   0.449283757993216,   0.449283757993216,   0.219417649252501,  -0.219417649252501,  -0.449283757993216,  -0.449283757993216,  -0.219417649252501,],
                    [0.569359398342846,   0.402347308162278,  -0.402347308162278,  -0.569359398342846,  -0.083506045090284,   0.083506045090284,  -0.083506045090284,   0.083506045090284,],
                    [-0.083506045090284,   0.083506045090284,  -0.083506045090284,   0.083506045090284,   0.569359398342846,   0.402347308162278,  -0.402347308162278,  -0.569359398342846,],
                    [0.707106781186547,  -0.707106781186547,                   0,                   0,                   0,                   0,                   0,                   0,],
                    [0,                   0,                   0.707106781186547,  -0.707106781186547,                   0,                   0,                   0,                   0,],
                    [0,                   0,                   0,                   0,                   0.707106781186547,  -0.707106781186547,                   0,                   0,],
                    [0,                   0,                   0,                   0,                   0,                   0,                   0.707106781186547,  -0.707106781186547,]
                    ]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dct':
        t_forward = [[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
                      0.353553390593274, 0.353553390593274, 0.353553390593274],
                     [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064, -0.097545161008064,
                      -0.277785116509801, -0.415734806151273, -0.490392640201615],
                     [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643,
                      -0.191341716182545, 0.191341716182545, 0.461939766255643],
                     [0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801, 0.277785116509801,
                      0.490392640201615, 0.097545161008064, -0.415734806151273],
                     [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274,
                      -0.353553390593274, -0.353553390593274, 0.353553390593274],
                     [0.277785116509801, -0.490392640201615, 0.097545161008064, 0.415734806151273, -0.415734806151273,
                      -0.097545161008064, 0.490392640201615, -0.277785116509801],
                     [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545,
                      0.461939766255643, -0.461939766255643, 0.191341716182545],
                     [0.097545161008064, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615,
                      -0.415734806151273, 0.277785116509801, -0.097545161008064]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 11 and transform_type == 'dct':
        t_forward = [
        [0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274,   0.353553390593274],
        [0.490392640201615,   0.415734806151273,   0.277785116509801,   0.097545161008064,  -0.097545161008064,  -0.277785116509801,  -0.415734806151273,  -0.490392640201615],
        [0.461939766255643,   0.191341716182545,  -0.191341716182545,  -0.461939766255643,  -0.461939766255643,  -0.191341716182545,   0.191341716182545,   0.461939766255643],
        [0.415734806151273,  -0.097545161008064, -0.490392640201615, -0.277785116509801,   0.277785116509801,   0.490392640201615,   0.097545161008064,  -0.415734806151273],
        [0.353553390593274,  -0.353553390593274,  -0.353553390593274,   0.353553390593274,   0.353553390593274,  -0.353553390593274,  -0.353553390593274,   0.353553390593274],
        [0.277785116509801,  -0.490392640201615,   0.097545161008064,   0.415734806151273,  -0.415734806151273,  -0.097545161008064,   0.490392640201615,  -0.277785116509801],
        [0.191341716182545,  -0.461939766255643,   0.461939766255643,  -0.191341716182545,  -0.191341716182545,   0.461939766255643,  -0.461939766255643,   0.191341716182545],
        [0.097545161008064,  -0.277785116509801,   0.415734806151273,  -0.490392640201615,   0.490392640201615,  -0.415734806151273,   0.277785116509801,  -0.097545161008064]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dst':
        t_forward = [[0.161229841765317, 0.303012985114696, 0.408248290463863, 0.464242826880013, 0.464242826880013,
                      0.408248290463863, 0.303012985114696, 0.161229841765317],
                     [0.303012985114696, 0.464242826880013, 0.408248290463863, 0.161229841765317, -0.161229841765317,
                      -0.408248290463863, -0.464242826880013, -0.303012985114696],
                     [0.408248290463863, 0.408248290463863, 0, -0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, 0.408248290463863],
                     [0.464242826880013, 0.161229841765317, -0.408248290463863, -0.303012985114696, 0.303012985114696,
                      0.408248290463863, -0.161229841765317, -0.464242826880013],
                     [0.464242826880013, -0.161229841765317, -0.408248290463863, 0.303012985114696, 0.303012985114696,
                      -0.408248290463863, -0.161229841765317, 0.464242826880013],
                     [0.408248290463863, -0.408248290463863, 0, 0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, -0.408248290463863],
                     [0.303012985114696, -0.464242826880013, 0.408248290463863, -0.161229841765317, -0.161229841765317,
                      0.408248290463863, -0.464242826880013, 0.303012985114696],
                     [0.161229841765317, -0.303012985114696, 0.408248290463863, -0.464242826880013, 0.464242826880013,
                      -0.408248290463863, 0.303012985114696, -0.161229841765317]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif transform_type == 'dct':
        t_forward = dct(np.eye(n), norm='ortho')
    elif transform_type == 'eye':
        t_forward = np.eye(n)
    elif transform_type == 'dst':
        t_forward = dst(np.eye(n), norm='ortho')
    elif transform_type == 'DCrand':
        x = np.random.normal(n)
        x[:, 0] = np.ones(len(x[:, 0]))
        q, _, _ = np.linalg.qr(x)
        if q[0] < 0:
            q = -q

        t_forward = q.T

    elif pywt is not None:
        # a wavelet decomposition supported by PyWavelets
        # Set periodic boundary conditions, to preserve bi-orthogonality
        t_forward = np.zeros((n, n))

        for ii in range(n):
            temp = np.zeros(n)
            temp[0] = 1.0
            temp = np.roll(temp, (ii, dec_levels))
            tt = pywt.wavedec(temp, transform_type, mode='periodization', level=int(np.log2(n)))
            cc = np.hstack(tt)
            t_forward[:, ii] = cc

    else:
        raise ValueError("Transform of " + transform_type + "couldn't be found and PyWavelets couldn't be imported!")

    t_forward = np.array(t_forward)
    # Normalize the basis elements
    if not ((n == 8) and transform_type == 'bior1.5'):
        try:
            t_forward = (t_forward.T @ np.diag(np.sqrt(1. / sum(t_forward ** 2, 0)))).T
        except TypeError:  # t_forward was not an array...
            pass

    # Compute the inverse transform matrix
    try:
        t_inverse = np.linalg.inv(t_forward)
    except LinAlgError:
        t_inverse = np.array([[1]])

    return t_forward, t_inverse



def _estimate_parameters_for_psd(psd65_full: np.ndarray) -> (list, list, list, list):
    """
    Estimate BM3D parameters based on the PSD.
    :param psd65_full: input PSDs (65x65xn)
    :return: (lambda, mu, refiltering lambda, refiltering mu)
    """

    # Get the optimal parameters and matching features for a bunch of PSDs
    path = os.path.dirname(__file__)
    data = loadmat(os.path.join(path, 'param_matching_data.mat'))
    features = data['features']
    maxes = data['maxes']

    sz = 65
    data_sz = 500
    indices_to_take = [1, 3, 5, 7, 9, 12, 17, 22, 27, 32]

    llambda = []
    wielambda = []
    llambda2 = []
    wielambda2 = []

    # Get separate parameters for each PSD provided
    for psd_num in range(psd65_full.shape[2] if len(psd65_full.shape) > 2 else 1):

        if len(psd65_full.shape) > 2:
            psd65 = fftshift(psd65_full[:, :, psd_num], axes=(0, 1))
        else:
            psd65 = fftshift(psd65_full[:, :], axes=(0, 1))

        # Get features for this PSD
        pcaxa = _get_features(psd65, sz, indices_to_take)

        # Calculate distances to other PSDs
        mm = np.mean(features, 1)
        f2 = features - np.repeat(np.atleast_2d(mm).T, data_sz, axis=1)
        c = f2 @ f2.T
        c /= data_sz
        pcax2 = pcaxa.T - mm
        u, s, v = svd(c)
        f2 = u @ f2
        pcax2 = u @ pcax2
        f2 = f2 * np.repeat(np.atleast_2d(np.sqrt(s)).T, 500, axis=1)
        pcax2 = pcax2 * np.sqrt(s)

        diff_pcax = np.sqrt(np.sum(abs(f2 - np.repeat(np.atleast_2d(pcax2).T, data_sz, axis=1)) ** 2, 0))
        dff_i = np.argsort(diff_pcax)

        # Take 20 most similar PSDs into consideration
        count = 20
        diff_indices = dff_i[0:count]

        # Invert, smaller -> bigger weight
        diff_inv = 1. / (diff_pcax + EPS)
        diff_inv = diff_inv[diff_indices] / np.sum(diff_inv[diff_indices])

        # Weight
        param_idxs = np.sum(diff_inv * maxes[diff_indices, :].T, 1)

        lambdas = np.linspace(2.5, 4.5, 21)
        wielambdas = np.linspace(0.2, 4.2, 21)

        # Get parameters from indices
        # Interpolate lambdas and mu^2s from the list
        for ix in [0, 2]:
            param_idx = max(1, param_idxs[ix]) - 1
            param_idx2 = max(1, param_idxs[ix + 1]) - 1

            l1 = lambdas[int(np.floor(param_idx))]
            l2 = lambdas[int(min(np.ceil(param_idx), lambdas.size - 1))]

            w1 = wielambdas[int(np.floor(param_idx2))]
            w2 = wielambdas[int(min(np.ceil(param_idx2), wielambdas.size - 1))]

            param_smooth = param_idx - np.floor(param_idx)
            param_smooth2 = param_idx2 - np.floor(param_idx2)

            if ix == 0:
                llambda.append(l2 * param_smooth + l1 * (1 - param_smooth))
                wielambda.append(w2 * param_smooth2 + w1 * (1 - param_smooth2))
            elif ix == 2:
                llambda2.append(l2 * param_smooth + l1 * (1 - param_smooth))
                wielambda2.append(w2 * param_smooth2 + w1 * (1 - param_smooth2))

    return llambda, wielambda, llambda2, wielambda2


def _get_features(psd: np.ndarray, sz: int, indices_to_take: list) -> np.ndarray:
    """
    Calculate features for a PSD from integrals
    :param psd: The PSD to calculate features for.
    :param sz: Size of the PSD.
    :param indices_to_take: Indices from which to split the integrals.
    :return: array of features, length indices_to_take*2
    """
    int_rot, int_rot2 = _pcax(psd)
    f1 = np.zeros(len(indices_to_take) * 2)

    for ii in range(0, len(indices_to_take)):
        rang = indices_to_take[ii]
        if ii > 0:
            rang = [i for i in range(indices_to_take[ii - 1], rang)]
        else:
            rang -= 1
        rn = len(rang) if type(rang) == list else 1
        f1[ii] = np.sum(int_rot[np.array([np.ceil(sz / 2) + rang - 1], dtype=int)]) / rn
        f1[len(indices_to_take) + ii] = np.sum(int_rot2[np.array([np.ceil(sz / 2) + rang - 1], dtype=int)]) / rn
        pass

    return f1


def _pcax(psd: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate integrals through the principal axes of psd.
    :param psd: psd.
    :return: (intg1, intg2) : two integrals along the two axes.
    """
    n = psd.shape[0]
    [g2, g1] = np.meshgrid([i for i in range(1, n+1)], [i for i in range(1, n+1)])

    def trapz2d(tg2, tg1, p):
        return np.trapz(_trapz2(tg2, p, 1), tg1[:, 0], axis=0)

    p_n = psd / trapz2d(g2, g1, psd)

    m2 = trapz2d(g2, g1, p_n * g2)
    m1 = trapz2d(g2, g1, p_n * g1)
    c = np.zeros(4)

    q1 = [2, 1, 1, 0]
    q2 = [0, 1, 1, 2]

    for jj in [0, 1, 3]:
        c[jj] = np.squeeze(trapz2d(g2, g1, p_n * (g2 - m2) ** q1[jj] * (g1 - m1) ** q2[jj]))

    c[2] = c[1]
    c = c.reshape((2, 2))
    u, s, v = svd(c)

    n3 = 3 * n

    g2_n3, g1_n3 = np.meshgrid(np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2,
                               np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2)

    # Rotate PSDs and calculate integrals along the rotated PSDs
    theta = np.angle(u[0, 0] + 1j * u[0, 1])
    g2_rot = g2_n3[n:2*n, n:2*n] * np.cos(theta) - g1_n3[n:2*n, n:2*n] * np.sin(theta)
    g1_rot = g1_n3[n:2*n, n:2*n] * np.cos(theta) + g2_n3[n:2*n, n:2*n] * np.sin(theta)

    psd_rep = np.tile(psd, (3, 3))
    psd_rot = interpn((g2_n3[0, :], g2_n3[0, :]), psd_rep, (g1_rot, g2_rot))
    int_rot = _trapz2(g1, psd_rot, 0)

    theta2 = np.angle(u[1, 0] + 1j * u[1, 1])
    g2_rot = g2_n3[n:2*n, n:2*n] * np.cos(theta2) - g1_n3[n:2*n, n:2*n] * np.sin(theta2)
    g1_rot = g1_n3[n:2*n, n:2*n] * np.cos(theta2) + g2_n3[n:2*n, n:2*n] * np.sin(theta2)
    psd_rot2 = interpn((g2_n3[0, :], g2_n3[0, :]), psd_rep, (g1_rot, g2_rot))
    int_rot2 = _trapz2(g1, psd_rot2, 0)

    return int_rot, int_rot2


def _trapz2(x: np.ndarray, y: np.ndarray, dimm: int) -> np.ndarray:
    """
    Calculate the integals of an 2-D array along specified dimension
    :param x: values of x
    :param y: values of y
    :param dimm: 1 or 0
    :return: integrals along the axis
    """
    if dimm == 1:
        intg = np.sum((y[:, 1:] + y[:, 0:-1]) / 2. * (x[:, 1:] - x[:, 0:-1]), axis=1)
    else:
        intg = np.sum((y[1:, :] + y[0:-1, :]) / 2. * (x[1:, :] - x[0:-1, :]), axis=0)
    return intg


def _get_kernel_from_psd(sigma_psd: Union[np.ndarray, float], single_dim_psd: bool = False) -> np.ndarray:
    """
    Calculate a correlation kernel from the input PSD / std through IFFT2
    :param sigma_psd: PSD or std / 3-d concatenation of such
    :param single_dim_psd: True if sigma_psd is a std
    :return: a correlation kernel
    """
    if single_dim_psd:
        return np.array(sigma_psd)

    sig = np.sqrt(sigma_psd / np.float(sigma_psd.shape[0] * sigma_psd.shape[1]))
    return fftshift(np.real(ifft2(sig, axes=(0, 1))), axes=(0, 1))


def _shrink_and_normalize_psd(temp_kernel: np.ndarray, new_size_2d: tuple) -> np.ndarray:
    """
    Calculate shrunk PSD from image-size, normalized, kernel.
    :param temp_kernel: Input kernel(s), MxNxC
    :param new_size_2d: new size, ignoring 3rd dimension
    :return: PSD of the normalized kernel
    """
    minus_size = np.array(np.ceil((np.array(temp_kernel.shape[:2]) - np.array(new_size_2d)) / 2), dtype=int)

    temp_kernel_shrunk = np.copy(temp_kernel[minus_size[0]:minus_size[0] + new_size_2d[0],
                                 minus_size[1]:minus_size[1] + new_size_2d[1]])

    for i in range(0, temp_kernel_shrunk.shape[2]):
        temp_kernel_shrunk[:, :, i] /= np.sqrt(np.sum(temp_kernel_shrunk[:, :, i] ** 2))

    return np.abs(fft2(temp_kernel_shrunk, shape=new_size_2d, axes=(0, 1))) ** 2 * new_size_2d[0] * new_size_2d[1]


def _process_psd(sigma_psd: Union[np.ndarray, float], z: np.ndarray,
                 single_dim_psd: bool, pad_size: tuple, profile: BM3DProfile)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Process input PSD for BM3D to acquire relevant inputs.
    :param sigma_psd: PSD (MxNxC) or a list of stds
    :param z: noisy image
    :param single_dim_psd: True if sigma_psd is a PSD (not std)
    :param pad_size: size to pad for refiltering
    :param profile: BM3DProfile used for this run
    :return: Tuple(sigma_psd2, psd_blur, psd_k)
            sigma_psd2 is equal to sigma_psd if refiltering is not used,
            otherwise it's the PSD in padded size
            psd_blur is equal to sigma_psd if Nf == 0 or single_dim_psd, otherwise it's a blurred PSD
            psd_k is the kernel used to blur the PSD (or [[[1]]])
    """
    temp_kernel = _get_kernel_from_psd(sigma_psd, single_dim_psd)

    auto_params = profile.lambda_thr3d is None or profile.lambda_wiener is None or \
        (profile.denoise_residual and (profile.lambda_thr3d2 is None or profile.lambda_wiener2 is None))

    if auto_params and not single_dim_psd:
        psd65 = _shrink_and_normalize_psd(temp_kernel, (65, 65))
        lambda_thr3d, lambda_wiener, lambda_thr3d2, lambda_wiener2 = _estimate_parameters_for_psd(psd65)
    else:
        lambda_thr3d = 3.0
        lambda_wiener = 0.4
        lambda_thr3d2 = 2.5
        lambda_wiener2 = 3.6

    # Create bigger PSD if needed
    if profile.denoise_residual and (pad_size[0] or pad_size[1]) and not single_dim_psd:

        pads_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0))
        temp_kernel = np.pad(temp_kernel, pads_width, 'constant')
        sigma_psd2 = abs(fft2(temp_kernel, axes=(0, 1))) ** 2 * z.shape[0] * z.shape[1]
    else:
        sigma_psd2 = sigma_psd

    # Ensure PSD resized to nf is usable
    if profile.nf > 0 and not single_dim_psd:

        sigma_psd_copy = _process_psd_for_nf(sigma_psd, None, profile)

        psd_k_sz = [1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[0] / profile.nf)),
                    1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[1] / profile.nf))]
        psd_k = gaussian_kernel([int(psd_k_sz[0]), int(psd_k_sz[1])],
                                1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[0] / profile.nf)) / 20,
                                1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[1] / profile.nf)) / 20)
        psd_k = psd_k / np.sum(psd_k)
        psd_k = np.array([psd_k]).transpose((1, 2, 0))

        psd_blur = correlate(sigma_psd_copy, psd_k, mode='wrap')
    else:
        psd_k = np.array([[[1]]])
        psd_blur = np.copy(sigma_psd)

    profile.lambda_thr3d = lambda_thr3d if profile.lambda_thr3d is None else profile.lambda_thr3d
    profile.lambda_wiener = lambda_wiener if profile.lambda_wiener is None else profile.lambda_wiener
    profile.lambda_thr3d2 = lambda_thr3d2 if profile.lambda_thr3d2 is None else profile.lambda_thr3d2
    profile.lambda_wiener2 = lambda_wiener2 if profile.lambda_wiener2 is None else profile.lambda_wiener2

    return sigma_psd2, psd_blur, psd_k


def _get_transforms(profile_obj: BM3DProfile, stage_ht: bool) -> (np.ndarray, np.ndarray, dict, dict, np.ndarray):
    """
    Get transform matrices used by BM3D.
    :param profile_obj: profile used by the execution.
    :param stage_ht: True if we are doing hard-thresholding with the results
    :return: t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, wwin2d
            (forward transform, inverse transform, 3rd dim forward transforms, 3rd dim inverse transforms,
            kaiser window for aggregation)
    """
    # get (normalized) forward and inverse transform matrices
    if stage_ht:
        t_forward, t_inverse = _get_transf_matrix_mod(profile_obj.bs_ht, profile_obj.transform_2d_ht_name,
                                                  profile_obj.dec_level, False) ## MOD
    else:
        t_forward, t_inverse = _get_transf_matrix_mod(profile_obj.bs_wiener, profile_obj.transform_2d_wiener_name,
                                                  0, False) ## MOD

    if profile_obj.transform_3rd_dim_name == 'haar' or profile_obj.transform_3rd_dim_name[-3:] == '1.1':
        # If Haar is used in the 3-rd dimension, then a fast internal transform is used,
        # thus no need to generate transform matrices.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}
    else:
        # Create transform matrices. The transforms are later applied by
        # matrix-vector multiplication for the 1D case.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}

        rangemax = np.ceil(np.log2(np.max([profile_obj.max_3d_size_ht, profile_obj.max_3d_size_wiener]))) + 1
        for hpow in range(0, int(rangemax)):
            h = 2 ** hpow
            t_forward_3d, t_inverse_3d = _get_transf_matrix(h, profile_obj.transform_3rd_dim_name, 0, True)
            hadper_trans_single_den[h] = t_forward_3d
            inverse_hadper_trans_single_den[h] = t_inverse_3d.T

    # 2D Kaiser windows used in the aggregation of block-wise estimates
    if profile_obj.beta_wiener == 2 and profile_obj.beta == 2 and profile_obj.bs_wiener == 8 and profile_obj.bs_ht == 8:
        # hardcode the window function so that the signal processing toolbox is not needed by default
        wwin2d = [[0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924]]
    else:
        if stage_ht:
            # Kaiser window used in the aggregation of the HT part
            wwin2d = np.transpose([np.kaiser(profile_obj.bs_ht, profile_obj.beta)]) @ [
                np.kaiser(profile_obj.bs_ht, profile_obj.beta)]
        else:
            # Kaiser window used in the aggregation of the Wiener filt. part
            wwin2d = np.transpose([np.kaiser(profile_obj.bs_wiener, profile_obj.beta_wiener)]) @ [
                np.kaiser(profile_obj.bs_wiener, profile_obj.beta_wiener)]

    wwin2d = np.array(wwin2d)
    return t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, wwin2d


def rgb_to(img: np.ndarray, colormode: str = 'YCbCr', inverse: bool = False,
           o_max: float = 0, o_min: float = 0)\
        -> (np.ndarray, float, float, float, np.ndarray):
    """
    Converts to normalized YCbCr or 'opp' (or back), returns normalization values needed for inverse
    :param img: image to transform (MxNx3)
    :param colormode: 'YCbCr' or 'opp'
    :param inverse: if True, do the inverse instead
    :param o_max: max value used for inverse scaling (returned by forward)
    :param o_min: min value used for inverse scaling (returned by forward)
    :return: (normalized+transformed image, o_max, o_min, scale used to multiply 1-D PSD, forward transform used)
    """
    if colormode == 'opp':
        # Forward
        a = np.array([[1/3, 1/3, 1/3], [0.5, 0, -0.5], [0.25, -0.5, 0.25]])
        # Inverse
        b = np.array([[1, 1, 2/3], [1, 0, -4/3], [1, -1, 2/3]])
    else:
        # YCbCr
        a = np.array([[0.299, 0.587, 0.114], [-0.168737, -0.331263, 0.5], [0.5, -0.418688, -0.081313]])
        b = np.array([[1.0000, 0.0000, 1.4020], [1.0000, -0.3441, -0.7141], [1.0000, 1.7720, 0.0000]])

    if inverse:
        # The inverse transform
        o = (img.reshape([img.shape[0] * img.shape[1], 3]) * (o_max - o_min) + o_min) @ b.T
        scale = None
    else:
        # The color transform
        o = img.reshape([img.shape[0] * img.shape[1], 3]) @ a.T
        o_max = np.max(o, axis=0)
        o_min = np.min(o, axis=0)
        o = (o - o_min) / (o_max - o_min)
        scale = np.sum(a.T ** 2, axis=0) / (o_max - o_min) ** 2

    return o.reshape([img.shape[0], img.shape[1], 3]), o_max, o_min, scale, a
