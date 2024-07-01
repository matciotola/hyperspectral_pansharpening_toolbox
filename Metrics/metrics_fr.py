import torch
from torch import nn


class LSR(nn.Module):
    def __init__(self):
        # Class initialization
        super(LSR, self).__init__()

    @staticmethod
    def forward(outputs, pan):
        pan = pan.double()
        outputs = outputs.double()

        pan_flatten = torch.flatten(pan, start_dim=-2).transpose(2, 1)
        fused_flatten = torch.flatten(outputs, start_dim=-2).transpose(2, 1)
        with torch.no_grad():
            alpha = (fused_flatten.pinverse() @ pan_flatten)[:, :, :, None]
        i_r = torch.sum(outputs * alpha, dim=1, keepdim=True)

        err_reg = pan - i_r

        cd = 1 - (torch.var(err_reg, dim=(1, 2, 3)) / torch.var(pan, dim=(1, 2, 3)))

        return cd


class D_sR(nn.Module):
    def __init__(self):
        super(D_sR, self).__init__()
        self.metric = LSR()

    def forward(self, outputs, pan):
        lsr = torch.mean(self.metric(outputs, pan))
        return 1.0 - lsr

