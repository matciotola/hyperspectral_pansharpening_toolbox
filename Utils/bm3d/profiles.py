"""
profiles.py
Contains definitions of BM3DStages and BM3DProfiles classes,
as well as several predefined BM3DProfile subclasses.
"""

import enum


class BM3DStages(enum.Enum):
    HARD_THRESHOLDING = 1
    WIENER_FILTERING = 2  # Pass a hard-thresholding estimate to the function instead of WIENER_FILTERING only
    ALL_STAGES = HARD_THRESHOLDING + WIENER_FILTERING


class BM3DProfile:
    """
    BM3DProfile object, containing the default settings for BM3D.
    Default values for our profile = 'np'
    """

    print_info = False

    # Transforms used
    transform_2d_ht_name = 'bior1.5'
    transform_2d_wiener_name = 'dct'
    transform_3rd_dim_name = 'haar'

    # -- Exact variances for correlated noise: --

    # Variance calculation parameters
    nf = 32  # domain size for FFT computations
    k = 4  # how many layers of var3D to calculate

    # Refiltering
    denoise_residual = False  # Perform residual thresholding and re-denoising
    residual_thr = 3  # Threshold for the residual HT (times sqrt(PSD))
    max_pad_size = None  # Maximum required pad size (= half of the kernel size), or 0 -> use image size

    # Block matching
    gamma = 4.0  # Block matching correction factor

    # -- Classic BM3D for correlated noise --

    # Hard-thresholding (HT) parameters:
    bs_ht = 8  # N1 x N1 is the block size used for the hard-thresholding (HT) filtering
    step_ht = 3  # sliding step to process every next reference block
    max_3d_size_ht = 16  # maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
    search_window_ht = 39  # side length of the search neighborhood for full-search block-matching (BM), must be odd
    tau_match = 3000  # threshold for the block-distance (d-distance)

    # None in these parameters results in automatic parameter selection for them
    lambda_thr3d = 2.7  # threshold parameter for the hard-thresholding in 3D transform domain
    lambda_wiener = None  # 1.0
    # Refilter
    lambda_thr3d2 = None
    lambda_wiener2 = None
    beta = 2.0  # parameter of the 2D Kaiser window used in the reconstruction

    # Wiener filtering parameters:
    bs_wiener = 8
    step_wiener = 3
    max_3d_size_wiener = 32
    search_window_wiener = 39
    tau_match_wiener = 400
    beta_wiener = 2.0
    dec_level = 0  # dec. levels of the dyadic wavelet 2D transform for blocks
    #  (0 means full decomposition, higher values decrease the dec. number)

    def get_block_size(self, mode: BM3DStages) -> int:
        """
        Get block size parameter.
        :param mode: BM3DStages enum value
        :return: block size
        """
        if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
            return self.bs_ht
        else:
            return self.bs_wiener

    def get_step_size(self, mode: BM3DStages) -> int:
        """
        Get step size parameter.
        :param mode: BM3DStages enum value
        :return: step size
        """
        if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
            return self.step_ht
        else:
            return self.step_wiener

    def get_max_3d_size(self, mode: BM3DStages) -> int:
        """
        Get maximum stack size in the 3rd dimension.
        :param mode: BM3DStages enum value
        :return: maximum stack size in the 3rd dimension
        """
        if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
            return self.max_3d_size_ht
        else:
            return self.max_3d_size_wiener

    def get_search_window(self, mode: BM3DStages) -> int:
        """
        Get search window size parameter.
        :param mode: BM3DStages enum value
        :return: search window size
        """
        if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
            return self.search_window_ht
        else:
            return self.search_window_wiener

    def get_block_threshold(self, mode: BM3DStages) -> int:
        """
        Get block matching threshold parameter.
        :param mode: BM3DStages enum value
        :return: block matching threshold
        """
        if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
            return self.tau_match
        else:
            return self.tau_match_wiener

    def __setattr__(self, attr, value):
        """
        Override __setattr__ to prevent adding new values (by typo).
        Raises AttributeError if a new attribute is added.
        :param attr: Attribute name to modify
        :param value: Value to modify
        """
        if not hasattr(self, attr):
            raise AttributeError("Unknown attribute name: " + attr)
        super(BM3DProfile, self).__setattr__(attr, value)



# class BM3DProfile:
#     """
#     BM3DProfile object, containing the default settings for BM3D.
#     Default values for our profile = 'np'
#     """
#
#     print_info = False
#
#     # Transforms used
#     transform_2d_ht_name = 'bior1.5'
#     transform_2d_wiener_name = 'dct'
#     transform_3rd_dim_name = 'haar'
#
#     # -- Exact variances for correlated noise: --
#
#     # Variance calculation parameters
#     nf = 32  # domain size for FFT computations
#     k = 4  # how many layers of var3D to calculate
#
#     # Refiltering
#     denoise_residual = False  # Perform residual thresholding and re-denoising
#     residual_thr = 3  # Threshold for the residual HT (times sqrt(PSD))
#     max_pad_size = None  # Maximum required pad size (= half of the kernel size), or 0 -> use image size
#
#     # Block matching
#     gamma = 4.0  # Block matching correction factor
#
#     # -- Classic BM3D for correlated noise --
#
#     # Hard-thresholding (HT) parameters:
#     bs_ht = 8  # N1 x N1 is the block size used for the hard-thresholding (HT) filtering
#     step_ht = 3  # sliding step to process every next reference block
#     max_3d_size_ht = 16  # maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
#     search_window_ht = 39  # side length of the search neighborhood for full-search block-matching (BM), must be odd
#     tau_match = 3000  # threshold for the block-distance (d-distance)
#
#     # None in these parameters results in automatic parameter selection for them
#     lambda_thr3d = None  # 2.7  # threshold parameter for the hard-thresholding in 3D transform domain
#     lambda_wiener = None  # 1.0
#     # Refilter
#     lambda_thr3d2 = None
#     lambda_wiener2 = None
#     beta = 2.0  # parameter of the 2D Kaiser window used in the reconstruction
#
#     # Wiener filtering parameters:
#     bs_wiener = 8
#     step_wiener = 3
#     max_3d_size_wiener = 32
#     search_window_wiener = 39
#     tau_match_wiener = 400
#     beta_wiener = 2.0
#     dec_level = 0  # dec. levels of the dyadic wavelet 2D transform for blocks
#     #  (0 means full decomposition, higher values decrease the dec. number)
#
#     def get_block_size(self, mode: BM3DStages) -> int:
#         """
#         Get block size parameter.
#         :param mode: BM3DStages enum value
#         :return: block size
#         """
#         if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
#             return self.bs_ht
#         else:
#             return self.bs_wiener
#
#     def get_step_size(self, mode: BM3DStages) -> int:
#         """
#         Get step size parameter.
#         :param mode: BM3DStages enum value
#         :return: step size
#         """
#         if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
#             return self.step_ht
#         else:
#             return self.step_wiener
#
#     def get_max_3d_size(self, mode: BM3DStages) -> int:
#         """
#         Get maximum stack size in the 3rd dimension.
#         :param mode: BM3DStages enum value
#         :return: maximum stack size in the 3rd dimension
#         """
#         if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
#             return self.max_3d_size_ht
#         else:
#             return self.max_3d_size_wiener
#
#     def get_search_window(self, mode: BM3DStages) -> int:
#         """
#         Get search window size parameter.
#         :param mode: BM3DStages enum value
#         :return: search window size
#         """
#         if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
#             return self.search_window_ht
#         else:
#             return self.search_window_wiener
#
#     def get_block_threshold(self, mode: BM3DStages) -> int:
#         """
#         Get block matching threshold parameter.
#         :param mode: BM3DStages enum value
#         :return: block matching threshold
#         """
#         if mode == BM3DStages.HARD_THRESHOLDING or mode == BM3DStages.ALL_STAGES:
#             return self.tau_match
#         else:
#             return self.tau_match_wiener
#
#     def __setattr__(self, attr, value):
#         """
#         Override __setattr__ to prevent adding new values (by typo).
#         Raises AttributeError if a new attribute is added.
#         :param attr: Attribute name to modify
#         :param value: Value to modify
#         """
#         if not hasattr(self, attr):
#             raise AttributeError("Unknown attribute name: " + attr)
#         super(BM3DProfile, self).__setattr__(attr, value)

class BM3DProfileRefilter(BM3DProfile):
    """
    Refiltering enabled
    """
    denoise_residual = True


"""
Profiles from old BM3D implementations:
"""


#  Profile 'vn' was proposed in
#  Y. Hou, C. Zhao, D. Yang, and Y. Cheng, 'Comment on "Image Denoising by Sparse 3D Transform-Domain
#  Collaborative Filtering"', accepted for publication, IEEE Trans. on Image Processing, July, 2010.
#  as a better alternative to that initially proposed in [1] (which is currently in profile 'vn_old')
class BM3DProfileVN(BM3DProfile):
    """
    'vn'
    """
    max_3d_size_ht = 32
    step_ht = 4

    bs_wiener = 11
    step_wiener = 6

    lambda_thr3d = 2.8
    tau_match_wiener = 3500

    search_window_wiener = 39


class BM3DProfileLC(BM3DProfile):
    """
    'lc'
    """
    step_ht = 6
    search_window_ht = 25
    step_wiener = 5
    max_3d_size_wiener = 16
    search_window_wiener = 25


class BM3DProfileVNOld(BM3DProfile):
    """
    'vn_old'
    """
    transform_2d_ht_name = 'dct'
    bs_ht = 12
    step_ht = 4

    bs_wiener = 11
    step_wiener = 6

    lambda_thr3d = 2.8
    tau_match_wiener = 3500
    tau_match = 5000

    search_window_wiener = 39


class BM3DProfileHigh(BM3DProfile):
    """
    'high'
    """
    dec_level = 1
    step_ht = 2
    step_wiener = 2
    lambda_thr3d = 2.5
    beta = 2.5
    beta_wiener = 1.5


class BM3DProfileDeb(BM3DProfile):
    """
    Parameters from old 'bm3d_deblurring' function.
    """
    transform_2d_ht_name = 'dst'
    lambda_thr3d = 2.9
    bs_wiener = 8
    step_wiener = 2
    max_3d_size_wiener = 16
    search_window_wiener = 39
    tau_match_wiener = 800
    beta_wiener = 0
