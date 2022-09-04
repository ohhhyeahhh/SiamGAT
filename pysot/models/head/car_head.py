import torch
from torch import nn
import math

from pysot.models.head.head_utils import ConvMixerBlock



class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x)))

        return logits, bbox_reg, centerness


class CARHead_CT(torch.nn.Module):
    def __init__(self, cfg, kernel=3, spatial_num=2):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead_CT, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES
        in_channels = cfg.TRAIN.CHANNEL_NUM

        cls_tower = []
        bbox_tower = []

        self.cls_spatial = ConvMixerBlock(in_channels, 3, depth=spatial_num)
        self.bbox_spatial = ConvMixerBlock(in_channels, 3, depth=spatial_num)

        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_x = self.cls_spatial(x)
        bbox_x = self.bbox_spatial(x)

        cls_tower = self.cls_tower(cls_x)
        bbox_tower = self.bbox_tower(bbox_x)

        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))

        return logits, bbox_reg, centerness




class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

