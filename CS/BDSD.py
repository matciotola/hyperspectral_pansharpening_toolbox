import torch
from torch.nn.functional import pad, interpolate

from Utils.imresize_bicubic import imresize
from Utils.spectral_tools import mtf, mtf_pan
from tqdm import tqdm
import cvxpy as cp

from Utils.dl_tools import normalize, denormalize

def lsqlin(C, d, A, b):
    A = A.squeeze(0).detach().cpu().numpy()
    b = b.squeeze().detach().cpu().numpy()
    C = C.squeeze(0).detach().cpu().numpy()
    d = d.squeeze().detach().cpu().numpy()

    x = cp.Variable(C.shape[1])

    constraints = [A @ x <= b]
    objective = cp.Minimize(cp.sum_squares(C @ x - d))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver='SCS', verbose=False, warm_start=False)

    x = torch.tensor(x.value)

    return x

def BDSD_PC(ordered_dict):

    ms, pan, ratio, sensor = ordered_dict.ms, ordered_dict.pan, ordered_dict.ratio, ordered_dict.sensor

    ms = normalize(ms)
    pan = normalize(pan)

    gt = imresize(ms, scale=1/ratio)
    ms_lr = mtf(gt, sensor, ratio)
    pan_lp = mtf_pan(pan, sensor, ratio)
    pan_lr = interpolate(pan_lp, scale_factor=1/ratio, mode='nearest-exact')

    fused = []
    for i in tqdm(range(ms.shape[1])):
        h1 = gt[:, i:i+1, :, :].transpose(2, 3).flatten(2).transpose(1, 2)
        h2 = ms_lr[:, i:i+1, :, :].transpose(2, 3).flatten(2).transpose(1, 2)
        h = torch.cat([pan_lr, ms_lr], dim= 1).transpose(2,3).flatten(2).transpose(1, 2)
        A = torch.eye(h.shape[2], dtype=h.dtype, device=h.device)[None, :, :].repeat(h.shape[0], 1, 1)
        A[:, 0, 0] = -1
        b = torch.zeros((1, h.shape[2], 1), dtype=h.dtype, device=h.device).repeat(h.shape[0], 1, 1)

        gamma = lsqlin(h, h1 - h2, A, b)
        fused.append(ms[:, i:i+1, :, :] + torch.reshape(torch.cat([pan, ms], dim=1).transpose(2,3).flatten(2).transpose(1, 2) @ gamma, (ms.shape[0], 1, ms.shape[3], ms.shape[2])).transpose(2, 3))
    fused = torch.cat(fused, dim=1)

    fused = denormalize(fused)

    return fused
