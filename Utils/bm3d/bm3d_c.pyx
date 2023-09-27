"""
Execute bm3d hard-thresholding and bm3d Wiener filtering stage calls with bm3d binaries.
"""

import numpy as np
from .profiles import BM3DProfile, BM3DStages
cimport numpy as np
from libc.stdlib cimport free

cdef extern from "bm3d_py.h":
    float* bm3d_threshold_colored_interface(double* z, float, int, int, int,
										int, int, float, int,
										float*, float*, float*, float*,
										float*, double*,
										float*, int, int, float, int,
										int*)

cdef extern from "bm3d_py.h":
    float* bm3d_wiener_colored_interface(double*, float*, int, int, int,
                                        int, int, float, int,
                                        float*, float*, float*, float*,
                                        float*, double*,
                                        float*, int, int, int, int*)


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
                                         psd.transpose(2, 1, 0).flatten()])
    else:# psd.ndim == 2:
        psd = np.concatenate([[1], [psd.shape[0]], [psd.shape[1]], psd.T.flatten()])
    return psd


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
    z = z.transpose(2, 1, 0).flatten()
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
        lambdas = np.array(lambdas)
        lambdas = np.ascontiguousarray(lambdas, dtype=np.float32)
    else:
        lambdas = None
        y_hat = np.ascontiguousarray(y_hat.transpose(2, 1, 0).flatten(), dtype=np.float32)

    z = np.ascontiguousarray(z, dtype=np.float64) # Makes a contiguous copy of the numpy array.

    psd = np.ascontiguousarray(psd, dtype=np.float32)
    t_fwd = np.ascontiguousarray(t_fwd, dtype=np.float32)
    t_inv = np.ascontiguousarray(t_inv, dtype=np.float32)
    t_fwd_3d_flat = np.ascontiguousarray(t_fwd_3d_flat, dtype=np.float32)
    t_inv_3d_flat = np.ascontiguousarray(t_inv_3d_flat, dtype=np.float32)

    wwin = np.ascontiguousarray(wwin, dtype=np.float64)

    pre_block_matches = np.ascontiguousarray(pre_block_matches, dtype=np.int32)

    cdef double[::1] c_z = z
    cdef double[::1] c_wwin = wwin
    cdef int[::1] c_pre_block_matches = pre_block_matches
    cdef float[::1] c_psd = psd
    cdef float[::1] c_t_fwd = t_fwd
    cdef float[::1] c_t_inv = t_inv
    cdef float[::1] c_t_fwd_3d_flat = t_fwd_3d_flat
    cdef float[::1] c_t_inv_3d_flat = t_inv_3d_flat

    # HT only
    cdef float[::1] c_lambdas = lambdas
    # Wiener only
    cdef float[::1] c_y_hat = y_hat

    cdef float c_thr3d = lambda_thr3d[0] if type(lambda_thr3d) is list else lambda_thr3d
    cdef int c_step = profile.get_step_size(mode)
    cdef int c_bs = profile.get_block_size(mode)
    cdef int c_max_3d_size = profile.get_max_3d_size(mode)
    cdef int c_sz1 = z_shape[0]
    cdef int c_sz2 = z_shape[1]
    cdef float c_thrclose = profile.get_block_threshold(mode) * np.power(profile.get_block_size(mode), 2) / (255 * 255)
    cdef int c_search_win_size = (profile.get_search_window(mode) - 1) / 2
    cdef int c_channel_count = channel_count
    cdef int c_nf = profile.nf
    cdef int c_k = profile.k
    cdef int c_gamma = profile.gamma

    cdef np.float32_t* ans
    if mode == BM3DStages.HARD_THRESHOLDING:
        ans = bm3d_threshold_colored_interface(&c_z[0], c_thr3d, c_step, c_bs, c_max_3d_size,
                                            c_sz1, c_sz2, c_thrclose, c_search_win_size,
                                            NULL if len(t_fwd) == 0 else &c_t_fwd[0],
                                            NULL if len(t_inv) == 0 else &c_t_inv[0],
                                            NULL if len(t_fwd_3d_flat) == 0 else &c_t_fwd_3d_flat[0],
                                            NULL if len(t_inv_3d_flat) == 0 else &c_t_inv_3d_flat[0],
                                            &c_lambdas[0], &c_wwin[0], &c_psd[0],
                                            c_nf, c_k, c_gamma,
                                            c_channel_count, &c_pre_block_matches[0])

    else:
        ans = bm3d_wiener_colored_interface(&c_z[0], &c_y_hat[0], c_step, c_bs, c_max_3d_size,
                                            c_sz1, c_sz2, c_thrclose, c_search_win_size,
                                            NULL if len(t_fwd) == 0 else &c_t_fwd[0],
                                            NULL if len(t_inv) == 0 else &c_t_inv[0],
                                            NULL if len(t_fwd_3d_flat) == 0 else &c_t_fwd_3d_flat[0],
                                            NULL if len(t_inv_3d_flat) == 0 else &c_t_inv_3d_flat[0],
                                            NULL, &c_wwin[0], &c_psd[0], c_nf, c_k,
                                            c_channel_count, &c_pre_block_matches[0])


    cdef int* ans_int = <int*>ans;

    # Read result image data
    cdef np.float32_t[:] view = <np.float32_t[:z.size]> ans

    # Read block matching data
    cdef int[:] block_view = <int[:z.size + (ans_int[z.size])]> ans_int

    numpy_arr = np.copy(np.asarray(view))
    numpy_arr_bm = np.array([])

    if pre_block_matches[0]:
        numpy_arr_bm = np.copy(np.asarray(block_view[z.size:], dtype=int))

    # This memory was allocated inside the call
    free(ans)
    return numpy_arr.reshape((z_shape[2], z_shape[1], z_shape[0])).transpose((2, 1, 0)), numpy_arr_bm


