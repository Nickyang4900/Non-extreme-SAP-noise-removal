import torch.nn as nn

class Net_16(nn.Module):
    def __init__(self):
        super(Net_16, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(15):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.Net_16 = nn.Sequential(*layers)
    def forward(self, x):
        out = self.Net_16(x)
        return out
