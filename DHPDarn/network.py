import torch
from torch import nn
import torch.nn.functional as func
import numpy as np
from .aux import get_lanczos_kernel


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

class Lanczos2Downsampler(nn.Module):

    def __init__(self, n_planes, factor, phase=0, preserve_size=False):
        super(Lanczos2Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        support = 2
        kernel_width = 4 * factor + 1

        self.kernel = get_lanczos_kernel(factor, phase, kernel_width, support)

        if factor % 2 == 0:
            p = 0
        else:
            p = 1

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=p, bias=False)

        kernel_torch = torch.from_numpy(self.kernel[None, None, :, :]).float().repeat(n_planes, n_planes, 1, 1)
        downsampler.weight.requires_grad = False
        downsampler.weight.data = kernel_torch

        self.downsampler = downsampler

        if preserve_size:

            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, x):
        if self.preserve_size:
            x = self.padding(x)
        return self.downsampler(x)


class CSA(nn.Module):
    def __init__(self, in_channels):
        super(CSA, self).__init__()
        self.in_channels = in_channels
        r = 16  # Downsampling ratio of the CA module

        # Input feature extraction
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)

        # CA
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=int(self.in_channels / r), kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=int(self.in_channels / r), out_channels=self.in_channels, kernel_size=1)

        # SA
        self.conv5 = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = self.conv2(func.relu(self.conv1(x)))
        M_CA = self.sigmoid(self.conv4(func.relu(self.conv3(self.gap(u)))))
        M_SA = self.sigmoid(self.conv5(u))
        U_CA = u * M_CA
        U_SA = u * M_SA
        out = U_CA + U_SA + x
        return out


class UpSample(nn.Module):
    def __init__(self, input_depth, num_channels_up, kernel_size, bias=True, pad_mode='reflect', up_mode='bilinear'):
        super(UpSample, self).__init__()
        self.conv1 = nn.Conv2d(input_depth, num_channels_up, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channels_up)

        self.conv2 = nn.Conv2d(num_channels_up, num_channels_up, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_channels_up)

        self.up = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = func.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = func.leaky_relu(x)
        x = self.up(x)

        return x


class DARN(nn.Module):
    def __init__(self, in_channels):
        super(DARN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.N_Filters = 64

        # FEN Layer
        self.FEN = nn.Conv2d(in_channels=self.in_channels + 1, out_channels=self.N_Filters, kernel_size=3, padding=1)
        # CSA RESBLOCKS
        self.CSA1 = CSA(in_channels=self.N_Filters)
        self.CSA2 = CSA(in_channels=self.N_Filters)
        self.CSA3 = CSA(in_channels=self.N_Filters)
        self.CSA4 = CSA(in_channels=self.N_Filters)
        # RNN layer
        self.RRN = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, prior, pan):

        # Concatenating the generated H_UP with P
        x = torch.cat((prior, pan), dim=1)

        # FEN
        x = self.FEN(x)

        # DARN
        x = self.CSA1(x)
        x = self.CSA2(x)
        x = self.CSA3(x)
        x = self.CSA4(x)

        # RRN
        x = self.RRN(x)

        # Final output
        output = x + prior

        return output


class DownSample(nn.Module):
    def __init__(self, input_depth, num_channels_down, kernel_size=3, bias=True, pad_mode='reflect'):
        super(DownSample, self).__init__()
        self.dw = Lanczos2Downsampler(input_depth, factor=2, preserve_size=True)
        self.bn1 = nn.BatchNorm2d(input_depth)
        self.conv2 = nn.Conv2d(input_depth, num_channels_down, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_channels_down)

    def forward(self, x):

        x = self.dw(x)
        x = self.bn1(x)
        x = func.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = func.leaky_relu(x)

        return x


class Skip (nn.Module):
    def __init__(self, input_depth, num_channel, kernel_size=1, pad_mode='reflect', bias=True):
        super(Skip, self).__init__()
        self.conv1 = nn.Conv2d(input_depth, num_channel, kernel_size, padding='same', padding_mode=pad_mode, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = func.leaky_relu(x)

        return x


class DHP (nn.Module):
    def __init__(self, num_inputs, num_channels_down=128, num_channels_up=128, num_channels_skip=4, kernel_size_down=3, kernel_size_up=3, kernel_size_skip=1):

        super(DHP, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, num_channels_down, kernel_size=3, padding=1, padding_mode='reflect')

        self.down_1 = DownSample(num_channels_down, num_channels_down, kernel_size_down)
        self.down_2 = DownSample(num_channels_down, num_channels_down, kernel_size_down)
        self.down_3 = DownSample(num_channels_down, num_channels_down, kernel_size_down)
        self.down_4 = DownSample(num_channels_down, num_channels_down, kernel_size_down)
        self.down_5 = DownSample(num_channels_down, num_channels_down, kernel_size_down)

        self.skip_1 = Skip(num_channels_down, num_channels_skip, kernel_size_skip)
        self.skip_2 = Skip(num_channels_down, num_channels_skip, kernel_size_skip)
        self.skip_3 = Skip(num_channels_down, num_channels_skip, kernel_size_skip)
        self.skip_4 = Skip(num_channels_down, num_channels_skip, kernel_size_skip)
        self.skip_5 = Skip(num_channels_down, num_channels_skip, kernel_size_skip)

        self.up_1 = UpSample(num_channels_down + num_channels_skip, num_channels_up, kernel_size_up)
        self.up_2 = UpSample(num_channels_up + num_channels_skip, num_channels_up, kernel_size_up)
        self.up_3 = UpSample(num_channels_up + num_channels_skip, num_channels_up, kernel_size_up)
        self.up_4 = UpSample(num_channels_up + num_channels_skip, num_channels_up, kernel_size_up)
        self.up_5 = UpSample(num_channels_up + num_channels_skip, num_channels_up, kernel_size_up)

        self.final_conv = nn.Conv2d(num_channels_up, num_inputs, 1, padding='same', padding_mode='reflect', bias=True)

    def forward(self, inputs):

        x = self.conv1(inputs)

        x1 = self.down_1(x)
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

class Downsampler(nn.Module):
    def __init__(self, kernel, ratio):
        super(Downsampler, self).__init__()

        self.conv = nn.Conv2d(kernel.shape[0], kernel.shape[0], kernel.shape[2], groups=kernel.shape[0], stride=1, padding='same', padding_mode='reflect', bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.data = kernel

        self.ratio = ratio

    def forward(self, x):
        x = self.conv(x)
        x = func.interpolate(x, scale_factor=1/self.ratio, mode='nearest-exact')

        return x

if __name__ == '__main__':

    inps = torch.rand(1, 91, 256, 256)

    model = DHP(91)

    out = model(inps)