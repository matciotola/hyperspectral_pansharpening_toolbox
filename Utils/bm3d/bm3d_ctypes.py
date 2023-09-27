"""
Execute bm3d hard-thresholding and bm3d Wiener filtering stage calls with bm3d binaries.
"""

import ctypes
from ctypes.util import find_library
import numpy as np
import os
from sys import platform
from .profiles import BM3DProfile, BM3DStages

ARGTYPES_THR = [ctypes.POINTER(ctypes.c_double),  # AVG
                ctypes.c_float,  # thr3D
                ctypes.c_int,  # P
                ctypes.c_int,  # nm1
                ctypes.c_int,  # nm2
                ctypes.c_int,  # sz1
                ctypes.c_int,  # sz2
                ctypes.c_float,  # thrClose
                ctypes.c_int,  # searchWinSize
                ctypes.POINTER(ctypes.c_float),  # fMatN1
                ctypes.POINTER(ctypes.c_float),  # iMatN1
                ctypes.POINTER(ctypes.c_float),  # arbMat
                ctypes.POINTER(ctypes.c_float),  # arbMatInv
                ctypes.POINTER(ctypes.c_float),  # sigmas
                ctypes.POINTER(ctypes.c_double),  # WIN
                ctypes.POINTER(ctypes.c_float),  # PSD
                ctypes.c_int,  # Nf
                ctypes.c_int,  # Kin
                ctypes.c_float,  # gamma
                ctypes.c_int,  # Channel count
                ctypes.POINTER(ctypes.c_int)  # Blockmatch info
                ]

ARGTYPES_WIE = [ctypes.POINTER(ctypes.c_double),  # Bn
                ctypes.POINTER(ctypes.c_float),  # AVG
                ctypes.c_int,  # P
                ctypes.c_int,  # nm1
                ctypes.c_int,  # nm2
                ctypes.c_int,  # sz1
                ctypes.c_int,  # sz2
                ctypes.c_float,  # thrClose
                ctypes.c_int,  # searchWinSize
                ctypes.POINTER(ctypes.c_float),  # fMatN1
                ctypes.POINTER(ctypes.c_float),  # iMatN1
                ctypes.POINTER(ctypes.c_float),  # arbMat
                ctypes.POINTER(ctypes.c_float),  # arbMatInv
                ctypes.POINTER(ctypes.c_float),  # sigmas
                ctypes.POINTER(ctypes.c_double),  # WIN
                ctypes.POINTER(ctypes.c_float),  # PSD
                ctypes.c_int,  # Nf
                ctypes.c_int,  # Kin
                ctypes.c_int,  # Channel count
                ctypes.POINTER(ctypes.c_int)  # Blockmatch info
                ]


def get_dll_names() -> tuple:
    """
    Generate shared library names to fetch.
    :return: thr library name, wiener library name
    """
    path = os.path.dirname(__file__)
    if platform == "darwin":
        dll_names = ["bm3d_thr_mac", "bm3d_wie_mac"]
    elif platform == "win32":
        dll_names = ["bm3d_thr_win", "bm3d_wie_win"]
    elif platform == "linux":
        dll_names = ["bm3d_thr", "bm3d_wie"]
    else:
        # Presume linux anyway ...
        dll_names = ["bm3d_thr", "bm3d_wie"]

    if platform != "win32" and find_library("openblas") is None:
        raise Exception("OpenBLAS library not found!")

    return os.path.join(path, dll_names[0]) + ".so", os.path.join(path, dll_names[1]) + ".so"


def llen(x) -> int:
    """
    Calculate the length of input if list, else give 1
    :param x: list or non-container type
    :return: length or 1
    """
    return len(x) if type(x) is list else 1


def flatten_transf(transf_dict: dict) -> np.ndarray:
    """
    Flatten the stack transforms computed by __get_transforms to format used by the binary.
    :param transf_dict: a stack transform dict returned by __get_transforms
    :return: 1-d list of transforms
    """
    total_list = []
    for key in sorted(transf_dict):
        flattened = list(transf_dict[key].flatten())
        total_list += flattened
    total_list = np.array(total_list)
    return total_list


def format_psd(psd: np.ndarray, single_dim_psd: bool) -> np.ndarray:
    """
    Format PSD in the format used by the binary file.
    :param psd: PSD matching the noisy image, or a (list of) noise standard deviation(s)
    :param single_dim_psd: True if giving stds, False if giving full PSDs
    :return: np.ndarray of concatenated PSD dimension data
    """
    if single_dim_psd:
        psd = np.concatenate([[0, psd.size], psd.flatten()])
    elif psd.ndim == 3:
        psd = np.concatenate([[psd.shape[2]], [psd.shape[0]], [psd.shape[1]],
                              psd.transpose((2, 1, 0)).flatten()])
    else:  # psd.ndim == 2:
        psd = np.concatenate([[1], [psd.shape[0]], [psd.shape[1]], psd.T.flatten()])
    return psd


def conv_to_array(pyarr, ctype=ctypes.c_float):
    return (ctype * len(pyarr))(*pyarr)


def bm3d_step(mode: BM3DStages, z: np.ndarray, psd: np.ndarray, single_dim_psd: bool, profile: BM3DProfile,
              t_fwd: np.ndarray, t_inv: np.ndarray, t_fwd_3d: dict, t_inv_3d: dict,
              wwin: np.ndarray, channel_count: int, pre_block_matches: np.ndarray,
              refiltering=False, y_hat=None) -> (np.array, np.array):
    """
    Perform either BM3D Hard-thresholding or Wiener step through an external library call.
    :param mode: BM3DStages.HARD_THRESHOLDING or BM3DStages.WIENER_FILTERING
    :param z: The noisy image, np.ndarray of MxNxChannels
    :param psd: PSD matching the noisy image, or a (list of) noise standard deviation(s)
    :param single_dim_psd: True if giving stds, False if giving full PSDs
    :param profile: BM3DProfile object containing the chosen parameters.
    :param t_fwd: forward transform for 2-D transform
    :param t_inv: inverse transform for 2-D transform
    :param t_fwd_3d: 1-D stack transform
    :param t_inv_3d: 1-D stack transform
    :param wwin: Aggregation window
    :param channel_count: number of channels of z
    :param pre_block_matches: [0], [1] or block matches array
    :param refiltering: True if this is the refiltering step
    :param y_hat: previous estimate, same size as z, only for Wiener
    :return: tuple (estimate, blockmatches), where estimate is the same size as z, and blockmatches either
             empty or containing the blockmatch data if pre_block_matches was [1]
    """
    z_shape = z.shape
    z = z.transpose((2, 1, 0)).flatten()
    t_inv = t_inv.T.flatten()
    t_fwd = t_fwd.T.flatten()
    t_inv_3d_flat = flatten_transf(t_inv_3d)
    t_fwd_3d_flat = flatten_transf(t_fwd_3d)
    wwin = wwin.T.flatten()

    psd = format_psd(psd, single_dim_psd)

    # HT only
    lambda_thr3d = profile.lambda_thr3d if not refiltering else profile.lambda_thr3d2

    if mode == BM3DStages.HARD_THRESHOLDING:
        lambdas = [llen(lambda_thr3d)] + lambda_thr3d if llen(lambda_thr3d) > 1 else [0]
        c_lambdas = conv_to_array(np.ascontiguousarray(np.array(lambdas), dtype=np.float32))
        c_y_hat = None
    else:
        c_lambdas = None
        c_y_hat = conv_to_array(np.ascontiguousarray(y_hat.transpose(2, 1, 0).flatten(), dtype=np.float32))

    c_z = conv_to_array(np.ascontiguousarray(z, dtype=np.float64), ctype=ctypes.c_double)

    c_psd = conv_to_array(np.ascontiguousarray(psd, dtype=np.float32))
    c_t_fwd = conv_to_array(np.ascontiguousarray(t_fwd, dtype=np.float32))
    c_t_inv = conv_to_array(np.ascontiguousarray(t_inv, dtype=np.float32))
    c_t_fwd_3d_flat = conv_to_array(np.ascontiguousarray(t_fwd_3d_flat, dtype=np.float32))
    c_t_inv_3d_flat = conv_to_array(np.ascontiguousarray(t_inv_3d_flat, dtype=np.float32))

    c_wwin = conv_to_array(np.ascontiguousarray(wwin, dtype=np.float64), ctype=ctypes.c_double)

    c_pre_block_matches = conv_to_array(np.ascontiguousarray(pre_block_matches, dtype=np.int32), ctype=ctypes.c_int)

    c_thr3d = ctypes.c_float(lambda_thr3d[0] if type(lambda_thr3d) is list else lambda_thr3d)
    c_step = ctypes.c_int(profile.get_step_size(mode))
    c_bs = ctypes.c_int(profile.get_block_size(mode))
    c_max_3d_size = ctypes.c_int(profile.get_max_3d_size(mode))
    c_sz1 = ctypes.c_int(z_shape[0])
    c_sz2 = ctypes.c_int(z_shape[1])
    c_thrclose = ctypes.c_float(profile.get_block_threshold(mode) *
                                np.power(profile.get_block_size(mode), 2) / (255 * 255))
    c_search_win_size = ctypes.c_int((profile.get_search_window(mode) - 1) // 2)
    c_channel_count = ctypes.c_int(channel_count)
    c_nf = ctypes.c_int(profile.nf)
    c_k = ctypes.c_int(profile.k)
    c_gamma = ctypes.c_float(profile.gamma)

    dll = ctypes.CDLL(get_dll_names()[0 if mode == BM3DStages.HARD_THRESHOLDING else 1])

    if mode == BM3DStages.HARD_THRESHOLDING:
        func = dll.bm3d_threshold_colored_interface
        func.argtypes = ARGTYPES_THR
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

        ans = func(c_z, c_thr3d, c_step, c_bs, c_max_3d_size,
                   c_sz1, c_sz2, c_thrclose, c_search_win_size,
                   None if len(t_fwd) == 0 else c_t_fwd,
                   None if len(t_inv) == 0 else c_t_inv,
                   None if len(t_fwd_3d_flat) == 0 else c_t_fwd_3d_flat,
                   None if len(t_inv_3d_flat) == 0 else c_t_inv_3d_flat,
                   c_lambdas, c_wwin, c_psd,
                   c_nf, c_k, c_gamma,
                   c_channel_count, c_pre_block_matches)

    else:
        func = dll.bm3d_wiener_colored_interface
        func.argtypes = ARGTYPES_WIE
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

        ans = func(c_z, c_y_hat, c_step, c_bs, c_max_3d_size,
                   c_sz1, c_sz2, c_thrclose, c_search_win_size,
                   None if len(t_fwd) == 0 else c_t_fwd,
                   None if len(t_inv) == 0 else c_t_inv,
                   None if len(t_fwd_3d_flat) == 0 else c_t_fwd_3d_flat,
                   None if len(t_inv_3d_flat) == 0 else c_t_inv_3d_flat,
                   c_lambdas, c_wwin, c_psd, c_nf, c_k,
                   c_channel_count, c_pre_block_matches)

    # Get the contents of the returned array possibly with some type of magic
    # (get the address, add the offset, and for some reason requires a cast to a float pointer after that)
    r0 = ans[0]
    holder = ctypes.POINTER(ctypes.c_float)
    returns = []

    width = z_shape[0]
    height = z_shape[1]
    for k in range(channel_count):
        ret_arr = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                ret_arr[i][j] = ctypes.cast(ctypes.addressof(r0)
                                            + ctypes.sizeof(ctypes.c_float)
                                            * (width * height * k + i * width + j), holder).contents.value
        returns.append(ret_arr.T)

    # Blockmatching stuff if needed
    bm_array = []
    if pre_block_matches[0] == 1:
        # Acquire the blockmatch data array...
        r0 = ans[0]
        holder = ctypes.POINTER(ctypes.c_int)
        # The size of image returns
        starting_point = ctypes.sizeof(ctypes.c_float) * (width * height * channel_count)

        # The first element of the BM contains its total size of int
        bm_array = np.zeros(ctypes.cast(ctypes.addressof(r0) + starting_point, holder).contents.value, dtype=np.intc)
        for i in range(bm_array.size):
            bm_array[i] = ctypes.cast(ctypes.addressof(r0) + starting_point + ctypes.sizeof(ctypes.c_int) * i,
                                      holder).contents.value

    return np.array(np.atleast_3d(returns)).transpose((1, 2, 0)), bm_array
