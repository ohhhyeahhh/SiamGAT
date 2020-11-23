# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamgat_googlenet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support googlenet;alexnet;
__C.BACKBONE.TYPE = 'googlenet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train backbone layers
__C.BACKBONE.TRAIN_LAYERS = []

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Crop_pad
__C.BACKBONE.CROP_PAD = 4

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# Backbone offset
__C.BACKBONE.OFFSET = 13

# Backbone stride
__C.BACKBONE.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

# SiamGAT
__C.TRAIN.ATTENTION = True

__C.TRACK.TYPE = 'SiamGATTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 287

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.OFFSET = 45

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.28, 0.16, 0.4]

__C.HP_SEARCH.GOT_10k = [0.7, 0.02, 0.35]

__C.HP_SEARCH.UAV123 = [0.24, 0.04, 0.04]

__C.HP_SEARCH.LaSOT = [0.35, 0.05, 0.18]
