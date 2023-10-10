import torch


def estimation_alpha(ms, pan):

    ms_f = torch.flatten(ms, start_dim=2).double()
    pan_f = torch.flatten(pan, start_dim=2).double()

    alpha = torch.linalg.lstsq(ms_f.transpose(1, 2), pan_f.transpose(1, 2)).solution
    return alpha[:,:,:, None].float()


def eps(X):
    from torch import floor, log2
    v = floor(log2(abs(X)))
    epsilon = 2**v * torch.finfo(X.dtype).eps
    return epsilon


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def regress(y, X):
    y = y.double()
    X = X.double()

    Q, R = torch.linalg.qr(X)
    """
    p = torch.sum(torch.abs(R.diagonal(dim1=1, dim2=2)) > max(n, ncolX)*eps(R[:,0,0]), dim=1)
    P = torch.bmm(torch.pinverse(X), Q)
    P = torch.bmm(P, R)
    _, perm = torch.max(P, 1)
    """
    b = torch.linalg.lstsq(R, torch.bmm(Q.transpose(1, 2), y)).solution

    return b


def mldivide(y, X):

    y = y.double()
    X = X.double()

    Q, R = torch.linalg.qr(X)
    """
    p = torch.sum(torch.abs(R.diagonal(dim1=1, dim2=2)) > max(n, ncolX)*eps(R[:,0,0]), dim=1)
    P = torch.bmm(torch.pinverse(X), Q)
    P = torch.bmm(P, R)
    _, perm = torch.max(P, 1)
    """
    b = torch.linalg.solve(R, Q.transpose(1, 2) @ y)
    return b