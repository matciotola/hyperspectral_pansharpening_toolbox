from torch.nn import functional as F
from Utils.spectral_tools import mtf

def downsample_protocol(bands_10, bands_20, bands_60, ratio):


    bands_10_lp = mtf(bands_10, 'S2_10', ratio)
    bands_20_lp = mtf(bands_20, 'S2_20', ratio)
    bands_60_lp = mtf(bands_60, 'S2_60', ratio)

    bands_10_lr = F.interpolate(bands_10_lp, scale_factor=1/ratio, mode='nearest-exact')
    bands_20_lr = F.interpolate(bands_20_lp, scale_factor=1/ratio, mode='nearest-exact')
    bands_60_lr = F.interpolate(bands_60_lp, scale_factor=1/ratio, mode='nearest-exact')

    return bands_10_lr, bands_20_lr, bands_60_lr