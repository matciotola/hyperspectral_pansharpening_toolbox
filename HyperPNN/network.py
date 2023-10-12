import torch
from torch import nn
from torch.nn import functional as func

class HyperPNN_model(nn.Module):
    def __init__(self, nbands, padding='same', mode='reflect', bias=True) -> None:
        super(HyperPNN_model, self).__init__()
        self.conv1 = nn.Conv2d(nbands, 64, 1, padding=padding, padding_mode=mode, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 1, padding=padding, padding_mode=mode, bias=bias)
        self.conv3 = nn.Conv2d(64+1, 64, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv6 = nn.Conv2d(64, 64, 1, padding=padding, padding_mode=mode, bias=bias)
        self.conv7 = nn.Conv2d(64, nbands, 1, padding=padding, padding_mode=mode, bias=bias)

    def forward(self, hs, pan):
        x = func.relu(self.conv1(hs))
        x = func.relu(self.conv2(x))

        y = torch.cat((pan, x), 1)
        y = func.relu(self.conv3(y))
        y = func.relu(self.conv4(y))
        y = func.relu(self.conv5(y))

        y = y + x

        y = func.relu(self.conv6(y))
        y = self.conv7(y)

        return y

