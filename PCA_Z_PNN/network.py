from torch import nn
from torch.nn import functional as func


class PCA_Z_PNN_model(nn.Module):
    def __init__(self, nbands, padding='same', padding_mode='reflect', bias=True) -> None:
        super(PCA_Z_PNN_model, self).__init__()
        self.conv1 = nn.Conv2d(nbands + 1, 48, 7, padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv2 = nn.Conv2d(48, 32, 5, padding=padding, padding_mode=padding_mode, bias=bias)
        self.conv3 = nn.Conv2d(32, nbands, 3, padding=padding, padding_mode=padding_mode, bias=bias)


    def forward(self, input):
        x = func.relu(self.conv1(input))
        x = func.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + input[:, :-1, :, :]
        return x