import torch
import numpy as np
from torch.nn.functional import conv2d
from torch.nn.functional import pad
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
        if ratio != 6: # TO DO: Delete for HyperSpectral
            h = np.clip(h, a_min=0, a_max=np.max(h))
            h = h / np.sum(h)
        else:
            h = np.real(h)
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

    if sensor == 'PRISMA':
        GNyq = [0.3] * nbands
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

    # wave = SWTForward(J=levels, wave=filters)
    # wave_img = wave(img)
    wave_img = ndwt2_working(img, level=levels, filters=tuple(starck_and_murtagh_filters()))
    for i in range(1, len(wave_img['dec'])):
        wave_img['dec'][i][:,:,:,:] = 0
        # wave_img[1][:, 1:, :, :] = 0

    # iwave = SWTInverse(wave=filters)
    # img_lr = iwave(wave_img)

    img_lr = indwt2_working(wave_img, 'c')

    img_lr = resize(img_lr, [img_lr.shape[-2] // ratio, img_lr.shape[-1] // ratio], interpolation=Inter.NEAREST_EXACT)


    return img_lr

def LPFilter(img, ratio):
    from math import log2, ceil
    from Utils.Wavelet.SWT import SWTForward, SWTInverse
    import pywt
    levels = ceil(log2(ratio))
    filters = pywt.Wavelet(filter_bank=tuple(starck_and_murtagh_filters()))

    # wave = SWTForward(J=levels, wave=filters)
    # wave_img = wave(img)
    wave_img = ndwt2_working(img, level=levels, filters=tuple(starck_and_murtagh_filters()))
    for i in range(1, len(wave_img['dec'])):
        wave_img['dec'][i][:,:,:,:] = 0
        # wave_img[1][:, 1:, :, :] = 0

    # iwave = SWTInverse(wave=filters)
    # img_lr = iwave(wave_img)

    img_lr = indwt2_working(wave_img, 'c')

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

    if sensor == 'PRISMA':
        GNyq = np.asarray([0.2]) # https://www.asi.it/wp-content/uploads/2021/02/PRISMA-Mission-Status-v1f-1.pdf
    elif sensor == 'WV3':
        GNyq = np.asarray([0.14])
    else:
        GNyq = np.asarray([0.15])

    fcut = 1 / np.double(ratio)

    alpha = np.sqrt(((kernel_size-1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = fspecial_gauss((kernel_size, kernel_size), alpha)
    Hd = H / np.max(H)
    h = np.kaiser(kernel_size, 0.5)
    h = np.real(fir_filter_wind(Hd, h))

    return h[:, :, None]


def ndwt2_working(X, level, filters):
    # Error handling


    LoD = [None, None]
    HiD = [None, None]
    LoR = [None, None]
    HiR = [None, None]


    LoD[0] = torch.tensor(filters[0])
    HiD[0] = torch.tensor(filters[1])
    LoR[0] = torch.tensor(filters[2])
    HiR[0] = torch.tensor(filters[3])
    LoD[1] = torch.tensor(filters[0])
    HiD[1] = torch.tensor(filters[1])
    LoR[1] = torch.tensor(filters[2])
    HiR[1] = torch.tensor(filters[3])

    dwtEXTM = 'sym'

    # Initialization
    if X.numel() == 0:
        return None

    sX = torch.tensor(X.size())
    X = X.double()
    sizes = torch.zeros(level + 1, len(sX), dtype=torch.long)
    sizes[level] = sX

    for k in range(1, level + 1):
        dec = decFUNC(X, LoD, HiD, dwtEXTM)
        X = dec[0][0]
        sizes[level - k] = torch.tensor(X.size())
        dec1 = []
        for j in range(len(dec[0])):
            for i in dec:
                dec1.append(i[j])

        dec = torch.cat(dec1, 1)


        if k > 1:
            cfs[0] = cfs[0][:, 1:, :, :]
            cfs.insert(0, dec)
        else:
            cfs = [dec]

    cfs1 = []

    for i in range(len(cfs)):
        for j in range(cfs[i].shape[1]):
            cfs1.append(cfs[i][:, j, None, :, :])

    cfs = cfs1

    WT = {
        'sizeINI': sX.tolist(),
        'level': level,
        'filters': {
            'LoD': LoD,
            'HiD': HiD,
            'LoR': LoR,
            'HiR': HiR
        },
        'mode': dwtEXTM,
        'dec': cfs,
        'sizes': sizes.tolist()
    }

    return WT


def decFUNC(X, LoD, HiD, dwtEXTM):
    dec = []
    permVect = []
    a_Lo, d_Hi = wdec1D(X, LoD[0], HiD[0], permVect, dwtEXTM)
    permVect = [0, 1, 3, 2]
    dec.append(wdec1D(a_Lo, LoD[1], HiD[1], permVect, dwtEXTM))
    dec.append(wdec1D(d_Hi, LoD[1], HiD[1], permVect, dwtEXTM))
    return dec


def wdec1D(X, Lo, Hi, perm, dwtEXTM):
    from torchvision.transforms.functional import pad as pad_vision

    if perm:
        X = X.permute(perm)

    sX = torch.tensor(X.size())

    if len(sX) < 3:
        sX = torch.cat((sX, torch.tensor([1])), dim=0)

    lf = len(Lo)
    lx = sX[-1]
    lc = lx + lf - 1

    if dwtEXTM == 'zpd':
        pass
    elif dwtEXTM in ['sym', 'symh']:
        X = pad_vision(X, (lf-1, 0, lf, 0), padding_mode='symmetric')
    elif dwtEXTM == 'sp0':
        X = torch.cat(
            (X[:, 0:lf - 1].unsqueeze(1).expand(-1, lf - 1, -1), X, X[:, -lx:].unsqueeze(1).expand(-1, lf - 1, -1)),
            dim=1)
    elif dwtEXTM in ['sp1', 'spd']:
        Z = torch.zeros(sX[0], sX[1] + 2 * lf - 2, sX[2])
        Z[:, lf:lf + lx, :] = X
        last = sX[1] + lf - 1
        for k in range(1, lf):
            Z[:, last + k, :] = 2 * Z[:, last + k - 1, :] - Z[:, last + k - 2, :]
            Z[:, lf - k, :] = 2 * Z[:, lf - k + 1, :] - Z[:, lf - k + 2, :]
        X = Z
    elif dwtEXTM == 'symw':
        X = torch.cat((X[:, lf - 1:1:-1, :], X, X[:, -2:-lf - 1:-1, :]), dim=1)
    elif dwtEXTM in ['asym', 'asymh']:
        X = torch.cat((-X[:, lf - 2::-1, :], X, -X[:, -1:-lf:-1, :]), dim=1)
    elif dwtEXTM == 'asymw':
        X = torch.cat((-X[:, lf - 1:1:-1, :], X, -X[:, -2:-lf - 1:-1, :]), dim=1)
    elif dwtEXTM == 'rndu':
        X = torch.cat((torch.randn(sX[0], lf - 1, sX[2]), X, torch.randn(sX[0], lf - 1, sX[2])), dim=1)
    elif dwtEXTM == 'rndn':
        X = torch.cat((torch.randn(sX[0], lf - 1, sX[2]), X, torch.randn(sX[0], lf - 1, sX[2])), dim=1)
    elif dwtEXTM == 'ppd':
        X = torch.cat((X[:, -lf + 1:, :], X, X[:, :lf - 1, :]), dim=1)
    elif dwtEXTM == 'per':
        if lx % 2 != 0:
            X = torch.cat((X, X[:, -1, :]), dim=1)
        X = torch.cat((X[:, -lf + 1:, :], X, X[:, :lf - 1, :]), dim=1)

    Lo = Lo[None, None, None, :].type(X.dtype)
    Hi = Hi[None, None, None, :].type(X.dtype)

    L = torch.conv2d(X, Lo, padding=(0,Lo.shape[-1] - 1))
    H = torch.conv2d(X, Hi, padding=(0,Hi.shape[-1] - 1))

    if dwtEXTM != 'zpd':
        lenL = L.size(-1)
        first = lf - 1
        last = lenL - lf + 1
        L = L[:, :, :, first:last]
        H = H[:, :, :, first:last]
        lenL = L.size(-1)
        first = ((lenL - lc) // 2)
        last = first + lc
        L = L[:, :, :, first:last]
        H = H[:, :, :, first:last]

    if dwtEXTM == 'per':
        first = 0
        last = lx
        L = L[:, :, :, first:last]
        H = H[:, :, :, first:last]

    if perm:
        L = L.permute(perm)
        H = H.permute(perm)

    return L, H


def indwt2_working(W, *args):
    nbIN = len(args)
    idxCFS = -1
    cfsFLAG = False

    if nbIN > 0:
        nbCELL = len(W['dec'])
        type = args[0]
        if not isinstance(type, str):
            raise ValueError("Invalid argument type")
        type = type.upper()
        cfsFLAG = type.startswith('C')
        if cfsFLAG:
            type = type[1:]

        idxCFS_mapping = {
            'D': 0, 'H': 0,
            'AA': 1, 'LL': 1, 'A': 1, 'L': 1,
            'AD': 2, 'LH': 2,
            'DA': 3, 'HL': 3,
            'DD': 4, 'HH': 4
        }

        idxCFS = idxCFS_mapping.get(type, -1)

        if nbIN > 1:
            levREC = args[1]
        else:
            levREC = W['level']

        if idxCFS > 1:
            idxCFS = idxCFS + 3 * (W['level'] - levREC)
            if not cfsFLAG:
                for j in range(nbCELL):
                    if j != idxCFS:
                        W['dec'][j] = torch.zeros_like(W['dec'][j])
            else:
                X = W['dec'][idxCFS]
                return X
        elif idxCFS == 1:
            if cfsFLAG and levREC == W['level']:
                X = W['dec'][0]
                return X
            idxMinToKill = 1 + 3 * (W['level'] - levREC) + 1
            for j in range(idxMinToKill, nbCELL):
                W['dec'][j] = torch.zeros_like(W['dec'][j])
        elif idxCFS == 0:
            idxMaxToKill = 1 + 3 * (W['level'] - levREC)
            for j in range(1, idxMaxToKill + 1):
                W['dec'][j] = torch.zeros_like(W['dec'][j])

    Lo = W['filters']['LoR']
    Hi = W['filters']['HiR']
    dwtEXTM = W['mode']
    perFLAG = dwtEXTM == 'per'
    cfs = W['dec']
    sizes = W['sizes']
    level = W['level']

    maxloop = level
    if idxCFS == 1 and cfsFLAG:
        maxloop = level - levREC

    idxBeg = 0
    for k in range(maxloop):
        idxEnd = idxBeg + 3
        dec = cfs[idxBeg:idxEnd+1]
        sizerec = sizes[k+1]
        X = recFUNC(dec, sizerec, Lo, Hi, perFLAG)
        cfs[idxBeg:idxEnd] = [None] * 3
        cfs[idxEnd] = X
        idxBeg = idxEnd

    if abs(idxCFS) == 1 and not cfsFLAG and len(W['sizeINI']) == 3:
        # X = X.to(torch.uint8)
        pass

    return X


def recFUNC(dec, sINI, Lo, Hi, perFLAG):
    # Reconstruction
    perm = [0, 1, 3, 2]
    W = []
    for i in range(2):
        W.append(wrec1D(dec[i*2], Lo[1], perm, perFLAG) + wrec1D(dec[(i*2)+1], Hi[1], perm, perFLAG))

    X = (wrec1D(W[0], Lo[0], [], perFLAG) + wrec1D(W[1], Hi[0], [], perFLAG)) / 4

    # Extraction of the central part
    sREC = X.shape
    F_0 = (sREC[-2] - sINI[-2]) // 2
    F_1 = (sREC[-1] - sINI[-1]) // 2
    C_0 = (sREC[-2] - sINI[-2] + 1) // 2
    C_1 = (sREC[-1] - sINI[-1] + 1) // 2
    # F = (sREC - sINI) // 2
    # C = (sREC - sINI + 1) // 2
    X = X[:, :, F_0:X.shape[-2]-C_0, F_1:X.shape[-1]-C_1]

    return X


def wrec1D(X, F, perm, perFLAG):
    if perm:
        X = X.permute(perm)

    if perFLAG:
        nb = F.size(0) - 1
        X = torch.cat((X, X[:, :nb]), dim=1)

    X = torch.nn.functional.conv2d(X, F[None, None, None, :].type(X.dtype), padding=(0, F.shape[-1] - 1))

    if perm:
        X = X.permute(perm)

    return X


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

