# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.googlenet_dilation_v4 import Inception3_dilation_v4
from pysot.models.backbone.googlenet_modify import Inception3_modify
from pysot.models.backbone.googlenet import Inception3
from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'googlenet': Inception3,
              'googlenet_modify': Inception3_modify,
              'googlenet_dilation_v4': Inception3_dilation_v4,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
