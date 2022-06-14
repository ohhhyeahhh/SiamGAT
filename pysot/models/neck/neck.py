# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class GoogLeNetAdjustLayer(nn.Module):
    '''
    with mask: F.interpolate
    '''
    def __init__(self, in_channels, out_channels, crop_pad=0, kernel=1):
        super(GoogLeNetAdjustLayer, self).__init__()
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
            nn.BatchNorm2d(out_channels, eps=0.001),
        )
        self.crop_pad = crop_pad

    def forward(self, x):
        x = self.channel_reduce(x)

        if x.shape[-1] > 25 and self.crop_pad > 0:
            crop_pad = self.crop_pad
            x = x[:, :, crop_pad:-crop_pad, crop_pad:-crop_pad]

        return x

