import torch
from torch.nn import functional as func


def TV(ordered_dict):
    yms, ypan, ratio, dataset = ordered_dict.ms_lr, ordered_dict.pan, ordered_dict.ratio, ordered_dict.dataset

    if dataset == 'WV3':

        alpha = 0.75
        lambda_ = 1e-3
        c = 8
        maxiter = 50
        w = torch.tensor([0.0657, 0.1012, 0.1537, 0.1473, 0.1245, 0.1545, 0.1338, 0.1192]).view(1, 8, 1, 1)

    elif dataset == 'PRISMA':
        alpha = 0.75
        lambda_ = 1e-3
        c = 8
        maxiter = 50
        w = torch.tensor([0.0015, 0.0014, 0.0016, 0.0016, 0.0018, 0.0017, 0.0019, 0.0019, 0.0021, 0.0023, 0.0026,
                          0.0028, 0.0029, 0.0030, 0.0030, 0.0031, 0.0032, 0.0031, 0.0033, 0.0034, 0.0035, 0.0036,
                          0.0036, 0.0037, 0.0036, 0.0037, 0.0038, 0.0040, 0.0044, 0.0048, 0.0056, 0.0063, 0.0069,
                          0.0076, 0.0083, 0.0083, 0.0091, 0.0086, 0.0088, 0.0093, 0.0089, 0.0093, 0.0094, 0.0095,
                          0.0095, 0.0096, 0.0100, 0.0102, 0.0100, 0.0096, 0.0109, 0.0103, 0.0099, 0.0089, 0.0088,
                          0.0094, 0.0077, 0.0079, 0.0080, 0.0082, 0.0082, 0.0083, 0.0085, 0.0086, 0.0088, 0.0089,
                          0.0090, 0.0089, 0.0091, 0.0090, 0.0088, 0.0069, 0.0080, 0.0081, 0.0086, 0.0089, 0.0087,
                          0.0088, 0.0088, 0.0089, 0.0093, 0.0099, 0.0100, 0.0095, 0.0090, 0.0087, 0.0087, 0.0082,
                          0.0085, 0.0051, 0.0056, 0.0061, 0.0064, 0.0065, 0.0065, 0.0067, 0.0068, 0.0069, 0.0071,
                          0.0072, 0.0073, 0.0073, 0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0073, 0.0072,
                          0.0071, 0.0072, 0.0071, 0.0070, 0.0068, 0.0068, 0.0066, 0.0064, 0.0065, 0.0048, 0.0052,
                          0.0053, 0.0049, 0.0053, 0.0048, 0.0050, 0.0049, 0.0048, 0.0048, 0.0048, 0.0047, 0.0047,
                          0.0048, 0.0049, 0.0048, 0.0048, 0.0049, 0.0049, 0.0049, 0.0050, 0.0049, 0.0050, 0.0051,
                          0.0050, 0.0050, 0.0049, 0.0049, 0.0047, 0.0047, 0.0046, 0.0045, 0.0045, 0.0044, 0.0042,
                          0.0042, 0.0038, 0.0035, 0.0038, 0.0040]).view(1, 159, 1, 1)

    z = torch.zeros((yms.shape[0], yms.shape[1] * 2, ypan.shape[2], ypan.shape[3]), dtype=yms.dtype, device=yms.device)
    x = torch.zeros((yms.shape[0], yms.shape[1], ypan.shape[2], ypan.shape[3]), dtype=yms.dtype, device=yms.device)

    for k in range(maxiter):
        b = compute_b(yms, ypan, ratio, x, alpha, w)
        z = znext(z,x,b,alpha,lambda_,c)
        x = xnext(z, b, alpha)

    fused = x
    return fused

def xnext(z1, b, alpha):
    x1 = (b - dt_z(z1)) / alpha
    return x1

def dt_z(z):
    return dx_t(z[:, :z.shape[1] // 2, :, :]) + dy_t(z[:, 4:z.shape[1] // 2 + 4, :, :])

def diff(img):
    img_pad_x = func.pad(img, (0, 1, 0, 0), mode='replicate')
    img_pad_y = func.pad(img, (0, 0, 0, 1), mode='replicate')

    diff_x = - img_pad_x[:, :, :, :-1] + img_pad_x[:, :, :, 1:]
    diff_y = - img_pad_y[:, :, :-1, :] + img_pad_y[:, :, 1:, :]

    return diff_x, diff_y


def compute_db(b):
    diff_x, diff_y = diff(b)
    db = torch.cat((diff_x, diff_y), dim=1)
    return db


def c_iddt_z(z, c):
    dtz = dx_t(z[:, :z.shape[1] // 2, :, :]) + dy_t(z[:, 4:z.shape[1] // 2 + 4, :, :])

    ddtz = compute_db(dtz)

    return ddtz


def dx_t(v):
    return dy_t(v.transpose(-2, -1)).transpose(-2, -1)


def dy_t(v):
    u0 = - v[:, :, 0:1, :]
    _, u1 = diff(v)
    u1 = - u1[:, :, :-1, :]
    u2 = v[:, :, -2:-1, :]
    dyt = torch.cat([u0, u1[:, :, :-1, :], u2], dim=2)
    return dyt


def znext(z0, x0, b, alpha, lambda_, c):

    diff_x, diff_y = diff(x0)
    w = (2 * alpha / lambda_ * torch.sqrt(diff_x ** 2 + diff_y ** 2)) + c
    w = w.repeat(1, 2, 1, 1)
    z1 = (compute_db(b) + c_iddt_z(z0, c)) / w

    return z1

def decimate(x, ratio):
    return func.interpolate(x, scale_factor=1/ratio, mode='bilinear', antialias=True)


def interpolate(x, ratio):
    return func.interpolate(x, scale_factor=ratio, mode='bilinear')


def compute_h(x, w, ratio):

    yms = decimate(x, ratio)
    ypan = torch.sum(w * x, dim=1, keepdim=True)

    return yms, ypan


def adjoint_h(yms, ypan, ratio, w):
    x = interpolate(yms, ratio) + w * ypan.repeat(1, yms.shape[1], 1, 1)
    return x


def compute_b(yms, ypan, ratio, xk, alpha, w):
    h_xms, h_xpan = compute_h(xk, w, ratio)
    b = alpha * xk + adjoint_h(yms - h_xms, ypan - h_xpan, ratio, w)
    return b


if __name__ == '__main__':
    import numpy as np
    from scipy import io
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_1_zoom.mat')

    ratio = 4
    sensor = 'WV3'

    ms_lr = torch.from_numpy(temp['I_MS_LR'].astype(np.float64)).permute(2, 0, 1).unsqueeze(0).double()
    pan = torch.from_numpy(temp['I_PAN'].astype(np.float64)).unsqueeze(0).unsqueeze(0).double()


    fused = TV(ms_lr, pan, ratio, sensor)
    plt.figure()
    plt.imshow(fused[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.show()
