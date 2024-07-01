import torch
from torch import nn
import torch.nn.functional as func
import math


def AddWeighted(img1, alpha, img2, beta, lambda_=0.0):
    img = (img1 * torch.multiply(torch.ones(img1.shape, dtype=torch.uint8, device=img1.device), alpha)) + (img2 * torch.multiply(torch.ones(img2.shape, dtype=torch.uint8, device=img1.device), beta)) + lambda_
    return img.round().float()

def laplacian_kernels():
    kernel = torch.tensor([[2., 0., 2.], [0., -8., 0.], [2., 0., 2.]], dtype=torch.float32)[None, None, :, :]
    return kernel


def sobel_kernels():
    kernel_x = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y


def prewitt_kernels():
    kernel_x = torch.tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y


def roberts_kernels():
    kernel_x = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y


def variance_scaling_initializer(tensor):

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal"):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            stddev = math.sqrt(scale) / .87962566103423978

        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                variance_scaling_initializer(m.weight)  # method 1: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()

        self.gaussian_filter = nn.Conv2d(1, 1, 3, padding='same', bias=False, padding_mode='reflect')

        self.sobel_x = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')
        self.sobel_y = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')

        self.prewitt_x = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')
        self.prewitt_y = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')

        self.roberts_x = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')
        self.roberts_y = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')

        self.laplacian = nn.Conv2d(1, 1, 3, bias=False, padding='same', padding_mode='reflect')

        self.sobel_x.weight.data, self.sobel_y.weight.data = sobel_kernels()
        self.prewitt_x.weight.data, self.prewitt_y.weight.data = prewitt_kernels()
        self.roberts_x.weight.data, self.roberts_y.weight.data = roberts_kernels()
        self.laplacian.weight.data = laplacian_kernels()

        self.gaussian_filter.weight.data = torch.tensor([[0.0625, 0.125, 0.0625],
                                                         [0.125, 0.25, 0.125],
                                                         [0.0625, 0.125, 0.0625]],
                                                        dtype=torch.float32)[None, None, :, :]

        self.gaussian_filter.weight.requires_grad = False
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        self.prewitt_x.weight.requires_grad = False
        self.prewitt_y.weight.requires_grad = False

        self.roberts_x.weight.requires_grad = False
        self.roberts_y.weight.requires_grad = False

        self.laplacian.weight.requires_grad = False

    def forward(self, pan):

            # pan_lp = filters.gaussian_blur2d(pan, (3, 3), (self.sigma, self.sigma), separable=False)
            pan_lp = self.gaussian_filter(pan)

            norm = torch.amax(pan_lp, dim=(1, 2, 3), keepdim=True)

            pan_lp = pan_lp * 255.0 / norm

            sobel_x = torch.clip(torch.abs(self.sobel_x(pan_lp)), 0, 255).round()
            sobel_y = torch.clip(torch.abs(self.sobel_y(pan_lp)), 0, 255).round()
            sobel = AddWeighted(sobel_x.int(), 0.5, sobel_y.int(), 0.5, 0.0)
            sobel = sobel * norm / 255.0

            pan_lp = pan_lp // 1.0

            prewitt_x = torch.clip(torch.abs(self.prewitt_x(pan_lp)), 0, 255).round()
            prewitt_y = torch.clip(torch.abs(self.prewitt_y(pan_lp)), 0, 255).round()
            prewitt = AddWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0.0)
            prewitt = prewitt * norm / 255.0

            roberts_x = self.roberts_x(pan_lp)
            roberts_y = self.roberts_y(pan_lp)

            roberts_x = torch.clip(torch.abs(roberts_x), 0, 255).round()
            roberts_y = torch.clip(torch.abs(roberts_y), 0, 255).round()
            roberts = AddWeighted(roberts_x, 0.5, roberts_y, 0.5, 0.0)
            roberts = roberts * norm / 255.0

            laplacian = self.laplacian(pan_lp)
            laplacian = torch.clip(torch.abs(laplacian), 0, 255).round()
            laplacian = laplacian * norm / 255.0

            pan_pad_x = func.pad(pan, (1, 0, 0, 0), mode='replicate')
            pan_pad_y = func.pad(pan, (0, 0, 1, 0), mode='replicate')

            diff_x = - pan_pad_x[:, :, :, :-1] + pan_pad_x[:, :, :, 1:]
            diff_y = - pan_pad_y[:, :, :-1, :] + pan_pad_y[:, :, 1:, :]

            x = torch.cat([diff_y, diff_x, roberts, prewitt, sobel, laplacian], dim=1)

            return x



class Block2(nn.Module):
    def __init__(self, channels, n_features=16):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(channels, n_features, 3, bias=True, padding='same')
        self.conv2 = nn.Conv2d(channels, n_features, 5, bias=True, padding='same')
        self.conv3 = nn.Conv2d(channels, n_features, 7, bias=True, padding='same')

        init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):

        x1 = func.relu(self.conv1(x))
        x2 = func.relu(self.conv2(x))
        x3 = func.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)

        return x


class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.conv1 = nn.Conv2d(48, 32, 3, bias=True, padding='same')
        self.conv2 = nn.Conv2d(32, 16, 3, bias=True, padding='same')
        self.conv3 = nn.Conv2d(16, 8, 3, bias=True, padding='same')
        self.conv4 = nn.Conv2d(8, 8, 3, bias=True, padding='same')

        init_weights(self.conv1, self.conv2, self.conv3, self.conv4)

    def forward(self, x):

        x1 = func.relu(self.conv1(x))
        x2 = func.relu(self.conv2(x1))
        x3 = func.relu(self.conv3(x2))
        x4 = func.relu(self.conv4(x3))
        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        init_weights(self.conv_du, self.avg_pool)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y


class Hyper_DSNet(nn.Module):
    def __init__(self, nbands):
        super(Hyper_DSNet, self).__init__()

        self.block1 = Block1()
        self.block2 = Block2(nbands + 7)
        self.block3 = Block3()
        self.CA = CALayer(nbands, 4, bias=True)
        self.convlast = nn.Conv2d(112, nbands, 1, bias=True)

    def forward(self, pan, ms, ms_lr):

        pan_edges = self.block1(pan)
        input1 = torch.cat((pan, pan_edges, ms), 1)
        input2 = self.block2(input1)
        output1 = self.block3(input2)
        output1 = self.convlast(output1)
        res = output1 * self.CA(ms_lr)
        output = res + ms

        return output


if __name__ == '__main__':
    from scipy import io
    import numpy as np
    temp = io.loadmat('/media/matteo/T7/Datasets/HyperSpectral/PRISMA/Test/RR1_Barcelona.mat')
    pan = temp['I_PAN'].astype(np.float32)

    pan = torch.from_numpy(pan[None, None, :, :]).repeat(4, 1, 1, 1)

    block1 = Block1()

    pan_edges = block1(pan)