import torch
import math
from torch import nn
import numpy as np
from torchvision.transforms import Pad


def normalize_block(im):
    m = torch.mean(im)
    s = torch.std(im)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_1d(onion1, onion2):
    bs, N = onion1.size()

    if N > 1:
        L = int(N / 2)
        a = onion1[:, :L]
        b = onion1[:, L:]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0] = 1
        b = b * neg
        c = onion2[:, :L]
        d = onion2[:, L:]
        d = d * neg

        if N == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_1d(a, c)
            ris2 = cayley_dickson_property_1d(d, torch.cat((torch.unsqueeze(b[:, 0], 1), -b[:, 1:]), dim=1))
            ris3 = cayley_dickson_property_1d(torch.cat((torch.unsqueeze(a[:, 0], 1), -a[:, 1:]), dim=1), d)
            ris4 = cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d(onion1, onion2):
    bs, dim3, _, _ = onion1.size()
    if dim3 > 1:
        L = int(dim3 / 2)

        a = onion1[:, 0:L, :, :]
        b = onion1[:, L:, :, :]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0, :, :] = 1
        b = b * neg
        c = onion2[:, 0:L, :, :]
        d = onion2[:, L:, :, :]
        d = d * neg
        if dim3 == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d, b * neg)
            ris3 = cayley_dickson_property_2d(a * neg, d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def q_index_metric(im1, im2, size):
    im1 = im1.double()
    im2 = im2.double()
    im2 = torch.cat((torch.unsqueeze(im2[:, 0, :, :], 1), -im2[:, 1:, :, :]), dim=1)
    batch_size, dim3, _, _ = im1.size()

    for bs in range(batch_size):
        for i in range(dim3):
            a1, s, t = normalize_block(im1[bs, i, :, :])
            im1[bs, i, :, :] = a1
            if s == 0:
                if i == 0:
                    im2[bs, i, :, :] = im2[bs, i, :, :] - s + 1
                else:
                    im2[bs, i, :, :] = -(-im2[bs, i, :, :] - s + 1)
            else:
                if i == 0:
                    im2[bs, i, :, :] = ((im2[bs, i, :, :] - s) / t) + 1
                else:
                    im2[bs, i, :, :] = -(((-im2[bs, i, :, :] - s) / t) + 1)

    m1 = torch.mean(im1, dim=(2, 3))
    m2 = torch.mean(im2, dim=(2, 3))
    mod_q1m = torch.sqrt(torch.sum(m1 ** 2, dim=1))
    mod_q2m = torch.sqrt(torch.sum(m2 ** 2, dim=1))

    mod_q1 = torch.sqrt(torch.sum(im1 ** 2, dim=1))
    mod_q2 = torch.sqrt(torch.sum(im2 ** 2, dim=1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = [size ** 2 / (size ** 2 - 1)] * batch_size
    temp = torch.tensor(temp, dtype=im1.dtype, device=im1.device)
    int1 = torch.clone(temp)
    int2 = torch.clone(temp)
    int3 = torch.clone(temp)
    int1 = int1 * torch.mean(mod_q1 ** 2)
    int2 = int2 * torch.mean(mod_q2 ** 2)
    int3 = int3 * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4
    if term3 == 0:
        q = torch.zeros((batch_size, dim3, 1, 1), device=im1.device, requires_grad=False)
        q[:, dim3 - 1, :, :] = mean_bias
    else:
        cbm = 2 / term3
        qu = cayley_dickson_property_2d(im1, im2)
        qm = cayley_dickson_property_1d(m1, m2)

        qv = (size ** 2) / (size ** 2 - 1) * torch.mean(qu, dim=(-2, -1))

        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q


class Q2n(nn.Module):
    def __init__(self, q_block_size=32, q_shift=32):
        super(Q2n, self).__init__()

        self.Q_block_size = q_block_size
        self.Q_shift = q_shift

    def forward(self, outputs, labels, channels=0):

        bs, dim3, dim1, dim2 = labels.size()
        _, _, ddim1, ddim2 = outputs.size()

        if channels == 0:
            channels = 2 ** math.ceil(math.log2(dim3))

        stepx = math.ceil(dim1 / self.Q_shift)
        stepy = math.ceil(dim2 / self.Q_shift)

        if stepy <= 0:
            stepy = 1
            stepx = 1

        est1 = (stepx - 1) * self.Q_shift + self.Q_block_size - dim1
        est2 = (stepy - 1) * self.Q_shift + self.Q_block_size - dim2

        outputs = torch.round(outputs)
        labels = torch.round(labels)

        if (est1 != 0) + (est2 != 0) > 0:
            padding = Pad((0, 0, est1, est2), padding_mode='symmetric')

            labels = padding(labels)
            outputs = padding(outputs)

        bs, dim3, dim1, dim2 = labels.size()

        if channels - math.log2(dim3) != 0:
            exp_difference = channels - dim3
            diff = torch.zeros((bs, exp_difference, dim1, dim2), device=outputs.device, requires_grad=False,
                               dtype=outputs.dtype)
            labels = torch.cat((labels, diff), dim=1)
            outputs = torch.cat((outputs, diff), dim=1)

        bs, dim3, dim1, dim2 = labels.size()
        values = torch.zeros((bs, dim3, stepx, stepy), device=outputs.device, requires_grad=False)

        for j in range(stepx):
            for i in range(stepy):
                o = q_index_metric(labels[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                                   i * self.Q_shift: i * self.Q_shift + self.Q_block_size],
                                   outputs[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                                   i * self.Q_shift: i * self.Q_shift + self.Q_block_size], self.Q_block_size)
                if len(o.shape) == 4:
                    o = torch.squeeze(o, dim=-1)
                    o = torch.squeeze(o, dim=-1)
                values.data[:, :, j, i] = o

        Q2n_index_map = torch.sqrt(torch.sum(values ** 2, dim=1))
        Q2n_index = torch.mean(Q2n_index_map)

        return Q2n_index, Q2n_index_map


class ERGAS(nn.Module):
    def __init__(self, ratio, reduction='mean'):
        super(ERGAS, self).__init__()
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, outputs, labels):
        mu = torch.mean(labels, dim=(2, 3)) ** 2
        nbands = labels.size(dim=1)
        error = torch.mean((outputs - labels) ** 2, dim=(2, 3))
        erg_all = 100 / self.ratio * torch.sqrt(error / mu)
        ergas_index = 100 / self.ratio * torch.sqrt(torch.sum(error / mu, dim=1) / nbands)
        if self.reduction == 'mean':
            ergas = torch.mean(ergas_index)
        else:
            ergas = torch.sum(ergas_index)

        return ergas, erg_all


class SAM(nn.Module):
    def __init__(self, reduction='mean'):
        super(SAM, self).__init__()
        self.reduction = reduction
        self.pi = np.pi

    def forward(self, outputs, labels):

        norm_outputs = torch.sum(outputs * outputs, dim=1)
        norm_labels = torch.sum(labels * labels, dim=1)
        scalar_product = torch.sum(outputs * labels, dim=1)
        norm_product = torch.sqrt(norm_outputs * norm_labels)
        scalar_product[norm_product == 0] = float('nan')
        norm_product[norm_product == 0] = float('nan')
        scalar_product = torch.flatten(scalar_product, 1, 2)
        norm_product = torch.flatten(norm_product, 1, 2)
        angle = torch.nansum(torch.acos(scalar_product / norm_product), dim=1) / norm_product.shape[1]
        angle = angle * 180 / self.pi
        return torch.mean(angle), angle


class Q(nn.Module):
    def __init__(self, nbands, block_size=32):
        super(Q, self).__init__()
        self.block_size = block_size
        self.N = block_size ** 2
        filter_shape = (nbands, 1, self.block_size, self.block_size)
        kernel = torch.ones(filter_shape, dtype=torch.float32)

        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)
        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False
        self.depthconv = self.depthconv.float()

    def forward(self, outputs, labels):
        outputs = outputs.float()
        labels = labels.float()
        outputs_sq = outputs ** 2
        labels_sq = labels ** 2
        outputs_labels = outputs * labels

        outputs_sum = self.depthconv(outputs)
        labels_sum = self.depthconv(labels)

        outputs_sq_sum = self.depthconv(outputs_sq)
        labels_sq_sum = self.depthconv(labels_sq)
        outputs_labels_sum = self.depthconv(outputs_labels)

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2
        numerator = 4 * (self.N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = self.N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)
        quality_map = torch.ones(denominator.size(), device=outputs.device)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = denominator != 0
        quality_map[index] = numerator[index] / denominator[index]
        quality = torch.mean(quality_map, dim=(2, 3))

        return quality
