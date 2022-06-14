import math
import torch
import torch.nn.functional as F
from torch import nn

class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size, depth):
        super().__init__()
        self.convmixer = nn.Sequential(
            *[nn.Sequential(
                # token mixing
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                # channel mixing
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            ) for i in range(depth)],
        )

    def forward(self, x):
        output = self.convmixer(x)
        return output


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)+x