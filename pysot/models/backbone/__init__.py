# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.googlenet import Inception3
from pysot.models.backbone.googlenet_ct import Inception3_ct
from pysot.models.backbone.googlenet_ou import Inception3_ou

BACKBONES = {
              # 'alexnetlegacy': alexnetlegacy,
              # 'mobilenetv2': mobilenetv2,
              # 'resnet18': resnet18,
              # 'resnet34': resnet34,
              # 'resnet50': resnet50,
              # 'alexnet': alexnet,
              'googlenet': Inception3,
              'googlenet_ct': Inception3_ct,
              'googlenet_ou': Inception3_ou,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
