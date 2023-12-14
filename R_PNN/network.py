from torch import nn
from torch.nn import functional as func

class R_PNN_model(nn.Module):
    def __init__(self, padding='valid', scope=6, bias=True) -> None:
        super(R_PNN_model, self).__init__()
        self.conv1 = nn.Conv2d(2, 48, 7, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(48, 32, 5, padding=padding, bias=bias)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=padding, bias=bias)

        self.scope = scope

    def forward(self, input):
        x = func.relu(self.conv1(input))
        x = func.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + input[:, :-1, self.scope:-self.scope, self.scope:-self.scope]
        return x

