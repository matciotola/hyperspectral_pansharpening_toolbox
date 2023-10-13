import torch
from torch import nn
from torch.nn import functional as func


class HSpeNet_model(nn.Module):
    def __init__(self, nbands, padding='same', mode='reflect', bias=True):
        super(HSpeNet_model, self).__init__()

        self.conv1_2 = nn.Conv2d(1, 16, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=padding, padding_mode=mode, bias=bias)

        self.conv1_1 = nn.Conv2d(nbands, 64, 1, padding=padding, padding_mode=mode, bias=bias)
        self.conv2_1 = nn.Conv2d(64, 64, 1, padding=padding, padding_mode=mode, bias=bias)

        self.conv3 = nn.Conv2d(64+16, 16, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv4 = nn.Conv2d(64+16+16, 16, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv5 = nn.Conv2d(64+16+16+16, 16, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv6 = nn.Conv2d(64+16+16+16+16, 16, 3, padding=padding, padding_mode=mode, bias=bias)
        self.conv7 = nn.Conv2d(64+16+16+16+16+16, 16, 3, padding=padding, padding_mode=mode, bias=bias)

        self.conv8 = nn.Conv2d(64+16+16+16+16+16+16, nbands, 1, padding=padding, padding_mode=mode, bias=bias)

    def forward(self, pan, hs):
        p = func.relu(self.conv1_2(pan))
        p = func.relu(self.conv2_2(p))
        p = pan - p     #o il contrario

        h = func.relu(self.conv1_1(hs))
        h = func.relu(self.conv2_1(h))

        y = torch.cat((p, h), 1)
        y1 = func.relu(self.conv3(y))

        y = torch.cat((y, y1), 1)
        y1 = func.relu(self.conv4(y))

        y = torch.cat((y, y1), 1)
        y1 = func.relu(self.conv5(y))

        y = torch.cat((y, y1), 1)
        y1 = func.relu(self.conv6(y))

        y = torch.cat((y, y1), 1)
        y1 = func.relu(self.conv7(y))

        y = torch.cat((y, y1), 1)
        y = self.conv8(y)

        y = y + hs

        return y