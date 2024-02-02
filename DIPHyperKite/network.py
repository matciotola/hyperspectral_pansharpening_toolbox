import torch
from torch import nn
import torch.nn.functional as func
import numpy as np


def adjust_shapes_and_cat(inputs):
    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]

    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

    return torch.cat(inputs_, dim=1)


class Skip (nn.Module):
    def __init__(self, input_depth, num_channel, kernel_size=1, pad_mode='reflect', bias=True):
        super(Skip, self).__init__()
        self.conv1 = nn.Conv2d(input_depth, num_channel, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = func.leaky_relu(x, negative_slope=0.02)

        return x


class DownSample(nn.Module):
    def __init__(self, input_depth, num_channels_down, kernel_size, bias=True, pad_mode='reflect'):
        super(DownSample, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(input_depth, num_channels_down, kernel_size, stride=2, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channels_down)
        self.conv2 = nn.Conv2d(num_channels_down, num_channels_down, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_channels_down)

    def forward(self, x):

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = func.leaky_relu(x, negative_slope=0.02)
        x = self.conv2(x)
        x = self.bn2(x)
        x = func.leaky_relu(x, negative_slope=0.02)

        return x


class UpSample(nn.Module):
    def __init__(self, input_depth, num_channels_up, kernel_size, bias=True, pad_mode='reflect', up_mode='bilinear'):
        super(UpSample, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_depth)
        self.conv1 = nn.Conv2d(input_depth, num_channels_up, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channels_up)

        self.conv2 = nn.Conv2d(num_channels_up, num_channels_up, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_channels_up)

        self.up = nn.Upsample(scale_factor=2, mode=up_mode)

        self.conv1x = nn.Conv2d(num_channels_up, num_channels_up, 1, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn1x = nn.BatchNorm2d(num_channels_up)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = func.leaky_relu(x, negative_slope=0.02)
        x = self.conv2(x)
        x = self.bn2(x)
        x = func.leaky_relu(x, negative_slope=0.02)
        x = self.up(x)

        x = self.conv1x(x)
        x = self.bn1x(x)
        x = func.leaky_relu(x, negative_slope=0.02)

        return x


class DIP (nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels_down_list=(128, 128, 128, 128, 128), num_channels_up_list=(128, 128, 128, 128, 128), num_channels_skip_list=(4, 4, 4, 4, 4), kernel_size_down=3, kernel_size_up=3, kernel_size_skip=1):

        super(DIP, self).__init__()

        self.down_1 = DownSample(num_inputs, num_channels_down_list[0], kernel_size_down)
        self.down_2 = DownSample(num_channels_down_list[0], num_channels_down_list[1], kernel_size_down)
        self.down_3 = DownSample(num_channels_down_list[1], num_channels_down_list[2], kernel_size_down)
        self.down_4 = DownSample(num_channels_down_list[2], num_channels_down_list[3], kernel_size_down)
        self.down_5 = DownSample(num_channels_down_list[3], num_channels_down_list[4], kernel_size_down)

        self.skip_1 = Skip(num_channels_down_list[0], num_channels_skip_list[0], kernel_size_skip)
        self.skip_2 = Skip(num_channels_down_list[0], num_channels_skip_list[1], kernel_size_skip)
        self.skip_3 = Skip(num_channels_down_list[1], num_channels_skip_list[2], kernel_size_skip)
        self.skip_4 = Skip(num_channels_down_list[2], num_channels_skip_list[3], kernel_size_skip)
        self.skip_5 = Skip(num_channels_down_list[3], num_channels_skip_list[4], kernel_size_skip)

        # self.up_0 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.up_1 = UpSample(num_channels_down_list[-1] + num_channels_skip_list[-1], num_channels_up_list[-1], kernel_size_up)
        self.up_2 = UpSample(num_channels_up_list[-1] + num_channels_skip_list[3], num_channels_up_list[-2], kernel_size_up)
        self.up_3 = UpSample(num_channels_up_list[-2] + num_channels_skip_list[2], num_channels_up_list[-3], kernel_size_up)
        self.up_4 = UpSample(num_channels_up_list[-3] + num_channels_skip_list[1], num_channels_up_list[-4], kernel_size_up)
        self.up_5 = UpSample(num_channels_up_list[-4] + num_channels_skip_list[-5], num_channels_up_list[-5], kernel_size_up)

        self.final_conv = nn.Conv2d(num_channels_up_list[-5], num_outputs, 1, padding='same', padding_mode='reflect', bias=True)

    def forward(self, inputs):

        x1 = self.down_1(inputs)
        s1 = self.skip_1(x1)

        x2 = self.down_2(x1)
        s2 = self.skip_2(x2)

        x3 = self.down_3(x2)
        s3 = self.skip_3(x3)

        x4 = self.down_4(x3)
        s4 = self.skip_4(x4)

        x5 = self.down_5(x4)
        s5 = self.skip_5(x5)

        x = self.up_1(adjust_shapes_and_cat([x5, s5]))
        x = self.up_2(adjust_shapes_and_cat([x, s4]))
        x = self.up_3(adjust_shapes_and_cat([x, s3]))
        x = self.up_4(adjust_shapes_and_cat([x, s2]))
        x = self.up_5(adjust_shapes_and_cat([x, s1]))
        x = self.final_conv(x)

        return x


class KiteNetwork(nn.Module):
    def __init__(self, nbands, scale):
        super(KiteNetwork, self).__init__()

        self.in_channels = nbands + 1
        self.out_channels = nbands
        self.scale = scale

        filters = [32, 64, 128]

        # ENCODER FILTERS
        self.encoder1 = nn.Conv2d(self.in_channels, filters[0], 3, stride=1, padding=1, padding_mode='reflect')
        self.ebn1 = nn.BatchNorm2d(filters[0])
        self.encoder2 = nn.Conv2d(filters[0], filters[1], 3, stride=1, padding=1, padding_mode='reflect')
        self.ebn2 = nn.BatchNorm2d(filters[1])
        self.encoder3 = nn.Conv2d(filters[1], filters[2], 3, stride=1, padding=1, padding_mode='reflect')
        self.ebn3 = nn.BatchNorm2d(filters[2])

        # BOTTELENECK FILTERS
        self.endec_conv = nn.Conv2d(filters[2], filters[2], 3, stride=1, padding=1, padding_mode='reflect')
        self.endec_bn = nn.BatchNorm2d(filters[2])

        # DECODER FILTERS
        self.decoder1 = nn.Conv2d(2 * filters[2], filters[1], 3, stride=1, padding=1, padding_mode='reflect')  # b, 1, 28, 28
        self.dbn1 = nn.BatchNorm2d(filters[1])
        self.decoder2 = nn.Conv2d(2 * filters[1], filters[0], 3, stride=1, padding=1, padding_mode='reflect')
        self.dbn2 = nn.BatchNorm2d(filters[0])
        self.decoder3 = nn.Conv2d(2 * filters[0], self.out_channels, 3, stride=1, padding=1, padding_mode='reflect')
        self.dbn3 = nn.BatchNorm2d(self.out_channels)

        # FINAL CONV LAYER
        self.final_conv = nn.Conv2d(self.out_channels, self.out_channels, 1, padding_mode='reflect')

        # RELU
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, pan, ms):

        # ms_up = func.interpolate(ms, scale_factor=(self.scale, self.scale), mode='bilinear')

        x = torch.cat((pan, ms), dim=1)

        # ENCODER
        t1 = self.relu(self.ebn1(self.encoder1(x)))

        out = func.interpolate(out, scale_factor=(2, 2), mode='bilinear')
        out = self.relu(self.ebn2(self.encoder2(out)))
        t2 = out
        out = func.interpolate(t1, scale_factor=(2, 2), mode='bilinear')


        out = func.interpolate(t2, scale_factor=(2, 2), mode='bilinear')
        t3 = self.relu(self.ebn3(self.encoder3(out)))

        # BOTTLENECK
        out = func.interpolate(t3, scale_factor=(2, 2), mode='bilinear')
        out = self.relu(self.endec_bn(self.endec_conv(out)))

        # DECODER
        out = func.max_pool2d(out, 2, 2)
        out = torch.cat((out, t3), dim=1)
        out = self.relu(self.dbn1(self.decoder1(out)))

        out = func.max_pool2d(out, 2, 2)
        out = torch.cat((out, t2), dim=1)
        out = self.relu(self.dbn2(self.decoder2(out)))

        out = func.max_pool2d(out, 2, 2)
        out = torch.cat((out, t1), dim=1)
        out = self.relu(self.dbn3(self.decoder3(out)))

        # OUTPUT CONV
        out = self.final_conv(out)

        # FINAL OUTPUT
        out = out + ms

        return out

class PanPredictionNetwork(nn.Module):
    def __init__(self, spectral_bands):
        super(PanPredictionNetwork, self).__init__()
        reduction = 2
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=spectral_bands, out_channels=spectral_bands // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=spectral_bands // reduction, out_channels=spectral_bands, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, principal_net_output):

        x = self.avgpool(principal_net_output)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.softmax(x)
        x = torch.sum(principal_net_output*x.expand_as(principal_net_output), dim=1, keepdim=True)

        return x


class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None,
                 preserve_size=False):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'

        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        if factor % 2 == 0:
            p = 0
        else:
            p = 1

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=p, bias=False)
        downsampler.weight.requires_grad = False
        downsampler.weight.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:

            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val


    else:
        assert False, 'wrong method name'

    kernel /= kernel.sum()

    return kernel

if __name__ == '__main__':

    net = DIP(32, 145)
    x = torch.randn(1, 32, 120, 120)
    y = net(x)