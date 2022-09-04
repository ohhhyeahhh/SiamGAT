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
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 287

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 4 # 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

__C.TRAIN.CHANNEL_NUM = 256

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'GOT', 'LaSOT', 'TrackingNet')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '/PATH/TO/VID'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = '/PATH/TO/YTBB'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '/PATH/TO/COCO'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = 50000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = '/PATH/TO/DET'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 50000

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = '/PATH/TO/GOT'
__C.DATASET.GOT.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 200000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = '/PATH/TO/LaSOT'
__C.DATASET.LaSOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 150000

__C.DATASET.TrackingNet = CN()
__C.DATASET.TrackingNet.ROOT = '/PATH/TO/TrackingNet'
__C.DATASET.TrackingNet.ANNO = 'training_dataset/trackingnet/train.json'
__C.DATASET.TrackingNet.FRAME_RANGE = 100
__C.DATASET.TrackingNet.NUM_USE = 350000

__C.DATASET.VIDEOS_PER_EPOCH = 800000

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
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "GoogLeNetAdjustLayer"

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

# __C.HP_SEARCH.OTB100 = [0.32, 0.3, 0.38]

__C.HP_SEARCH.GOT_10k = [0.7, 0.02, 0.35]

# __C.HP_SEARCH.GOT_10k = [0.9, 0.25, 0.35]

__C.HP_SEARCH.UAV123 = [0.24, 0.04, 0.04]

__C.HP_SEARCH.LaSOT = [0.35, 0.05, 0.18]

# __C.HP_SEARCH.LaSOT = [0.45, 0.05, 0.18]

# __C.HP_SEARCH.TrackingNet = [0.4, 0.05, 0.4]