import torch
import numpy as np
from torch.nn.functional import conv2d, pad
from torchvision.transforms import InterpolationMode as Inter

def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).
        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.
        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis=1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)
    return h

def fsamp2(hd):
    """
        Compute fir filter with window method
        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = np.real(h)

    return h

def fir_filter_wind(f1, f2):
    """
        Compute fir filter with window method
        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """

    hd = f1
    w1 = f2
    n = w1.shape[0]
    m = n
    t = np.arange(start=-(n-1)/2, stop=(n-1)/2 + 1) * 2/(n-1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 ** 2 + t2 ** 2)

    d = np.asarray(((t12 < t[0]) + (t12 > t[-1])).flatten()).nonzero()
    dd = (t12 < t[0]) + (t12 > t[-1])

    t12[dd] = 0

    w = np.interp(t12.flatten(),t, w1).reshape(t12.shape)
    w[dd] = 0
    h = fsamp2(hd) * w

    return h

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter
        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension
        """
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.
        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.
    """
    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)
        nyquist_freq = np.reshape(nyquist_freq, (1,nyquist_freq.shape[0]))

    nbands = nyquist_freq.shape[1]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[0, j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)
        h = np.real(fir_filter_wind(Hd, h))
        h = np.clip(h, a_min=0, a_max=np.max(h))
        h = h / np.sum(h)
        kernel[:, :, j] = h

    return kernel

def gen_mtf(ratio, sensor='none', kernel_size=41, nbands=3):
    """
        Compute the estimated MTF filter kernels for the supported satellites.
        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.
        """
    GNyq = []

    if sensor == 'S2-10':
        GNyq = [0.275, 0.28, 0.25, 0.24]
    elif sensor == 'S2-10-PAN':
        GNyq = [0.26125] * nbands
    elif sensor == 'S2-20':
        GNyq = [0.365, 0.33, 0.34, 0.32, 0.205, 0.235]
    elif sensor == 'S2-60':
        GNyq = [0.3175, 0.295, 0.30]
    elif sensor == 'S2-60_bis':
        GNyq = [0.3175, 0.295]
    elif sensor == 'WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315] ## TO REMOVE
    else:
        GNyq = [0.3] * nbands

    h = nyquist_filter_generator(GNyq, ratio, kernel_size)

    return h


def LPfilterGauss(img, ratio):
    GNyq = 0.3
    N = 41

    fcut = 1 / np.double(ratio)

    alpha = np.sqrt((N * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = fspecial_gauss((N, N), alpha)
    Hd = H / np.max(H)
    h = np.kaiser(N, 0.5)

    h = np.real(fir_filter_wind(Hd, h))[:, :, None]

    h = mtf_kernel_to_torch(h).repeat(img.shape[1], 1, 1, 1).type(img.dtype).to(img.device)

    I_PAN_LP = conv2d(pad(img, (h.shape[-2] // 2, h.shape[-2] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode='replicate'), h, padding='valid', groups=img.shape[1])
    # I_PAN_LP = ndimage.correlate(I_PAN, np.real(kernel), mode='nearest')

    I_Filtered = I_PAN_LP

    return I_Filtered


def mtf(img, sensor, ratio, mode='replicate'):
    h = gen_mtf(ratio, sensor, nbands=img.shape[1])

    h = mtf_kernel_to_torch(h).type(img.dtype).to(img.device)
    img_lp = conv2d(pad(img, (h.shape[-2] // 2, h.shape[-2] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode=mode), h, padding='valid', groups=img.shape[1])

    return img_lp

def mtf_pan(img, sensor, ratio, mode='replicate'):
    h = gen_mtf_pan(ratio, sensor)
    h = mtf_kernel_to_torch(h).type(img.dtype).to(img.device)
    img_lp = conv2d(pad(img, (h.shape[-2] // 2, h.shape[-2] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode=mode), h, padding='valid', groups=img.shape[1])

    return img_lp


def starck_and_murtagh_filters():

    from math import sqrt

    h1 = torch.tensor([1, 4, 6, 4, 1]) / 16
    h2 = torch.clone(h1)
    g = torch.zeros(5)
    g[2] = 1
    g1 = g - h1
    g2 = g + h1

    h1 = sqrt(2) * h1
    h2 = sqrt(2) * h2
    g1 = sqrt(2) * g1
    g2 = sqrt(2) * g2

    return h1.numpy(), g1.numpy(), h2.numpy(), g2.numpy()

# def LPFilterPlusDec(img, ratio):
#     from math import log2, ceil
#     from torchvision.transforms.functional import resize
#     img_n = torch.squeeze(img).numpy()
#     levels = ceil(log2(ratio))
#     filters = pywt.Wavelet(filter_bank=tuple(starck_and_murtagh_filters()))
#
#
#     (a, b), (c, d) = pywt.swt2(img_n, filters, level=levels)
#
#     b = np.asarray(b)
#     c[:,:] = 0
#     d = np.asarray(d)
#     b = list(np.zeros(b.shape))
#     d = list(np.zeros(d.shape))
#
#     coeff = ((a, b), (c, d))
#
#     img_lr = pywt.iswt2(coeff, filters)[None, None, :, :]
#
#     img_lr = torch.tensor(img_lr)
#
#     img_lr = resize(img_lr, [img_lr.shape[-2] // ratio, img_lr.shape[-1] // ratio], interpolation=Inter.NEAREST)
#
#
#     return img_lr


def LPFilterPlusDec(img, ratio):
    from math import log2, ceil
    from torchvision.transforms.functional import resize
    img_lr = []
    for i in range(img.shape[0]):

        img_n = torch.squeeze(img[i,:,:,:]).numpy()
        levels = ceil(log2(ratio))
        filters = pywt.Wavelet(filter_bank=tuple(starck_and_murtagh_filters()))


        (a, b), (c, d) = pywt.swt2(img_n, filters, level=levels)

        b = np.asarray(b)
        c[:,:] = 0
        d = np.asarray(d)
        b = list(np.zeros(b.shape))
        d = list(np.zeros(d.shape))

        coeff = ((a, b), (c, d))

        img_lr.append(torch.tensor(pywt.iswt2(coeff, filters)[None, None, :, :]))

    img_lr = torch.vstack(img_lr)

    img_lr = resize(img_lr, [img_lr.shape[-2] // ratio, img_lr.shape[-1] // ratio], interpolation=Inter.NEAREST)


    return img_lr


def LPFilterPlusDecTorch(img, ratio):
    from math import log2, ceil
    from torchvision.transforms.functional import resize
    from Utils.Wavelet.SWT import SWTForward, SWTInverse
    import pywt
    levels = ceil(log2(ratio))

    h1, g1, h2, g2 = starck_and_murtagh_filters()
    filter_bank = (h1.astype(np.float64), g1.astype(np.float64), h2.astype(np.float64), g2.astype(np.float64))
    filters = pywt.Wavelet(filter_bank=filter_bank)

    wave = SWTForward(J=levels, wave=filters)
    wave_img = wave(img)
    for i in range(levels):
        wave_img[i][:,1:,:,:] = 0
        # wave_img[1][:, 1:, :, :] = 0

    iwave = SWTInverse(wave=filters)
    img_lr = iwave(wave_img)

    img_lr = resize(img_lr, [img_lr.shape[-2] // ratio, img_lr.shape[-1] // ratio], interpolation=Inter.NEAREST)


    return img_lr

def LPFilter(img, ratio):
    from math import log2, ceil
    from Utils.Wavelet.SWT import SWTForward, SWTInverse
    import pywt
    levels = ceil(log2(ratio))
    filters = pywt.Wavelet(filter_bank=tuple(starck_and_murtagh_filters()))

    wave = SWTForward(J=levels, wave=filters)
    wave_img = wave(img)
    for i in range(levels):
        wave_img[i][:,1:,:,:] = 0
        # wave_img[1][:, 1:, :, :] = 0

    iwave = SWTInverse(wave=filters)
    img_lr = iwave(wave_img)

    return img_lr

def gen_mtf_pan(ratio, sensor, kernel_size=41):
    """
        Compute the estimated MTF filter kernels for the supported satellites.

        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.

        """
    GNyq = []

    if sensor == 'QB':
        GNyq = np.asarray([0.15])
    elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
        GNyq = np.asarray([0.17])
    elif (sensor == 'GeoEye1') or (sensor == 'GE1'):
        GNyq = np.asarray([0.16])
    elif sensor == 'WV2':
        GNyq = np.asarray([0.11])
    elif sensor == 'WV3':
        GNyq = np.asarray([0.14])
    else:
        GNyq = np.asarray([0.15])

    fcut = 1 / np.double(ratio)

    alpha = np.sqrt(((kernel_size) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = fspecial_gauss((kernel_size, kernel_size), alpha)
    Hd = H / np.max(H)
    h = np.kaiser(kernel_size, 0.5)
    h = np.real(fir_filter_wind(Hd, h))

    return h[:, :, None]



if __name__ == '__main__':

    import scipy.io as io
    import pywt
    from matplotlib import pyplot as plt

    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_3.mat')
    pan = torch.tensor(temp['I_PAN'].astype(np.float32))[None, None, :, :]
    pan_n = temp['I_PAN'].astype(np.float32)
    ratio = 4

    aaa = LPFilterPlusDecTorch(pan, ratio)

    plt.figure()
    plt.imshow(pan.numpy()[0,0,:,:])

    plt.figure()
    plt.imshow(aaa.numpy()[0,0,:,:])

