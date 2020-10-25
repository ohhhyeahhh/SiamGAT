# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

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

__C.TRAIN.LOC_WEIGHT = 2.0

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
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'GOT', 'LaSOT', 'TrackingNet')

# ############################SiamCAR_ORIGINAL#############################
# __C.DATASET.VID = CN()
# __C.DATASET.VID.ROOT = 'train_dataset/vid/crop511'          # VID dataset path
# __C.DATASET.VID.ANNO = 'train_dataset/vid/train.json'
# __C.DATASET.VID.FRAME_RANGE = 100
# __C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE
#
# __C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = 'train_dataset/yt_bb/crop511'  # YOUTUBEBB dataset path
# __C.DATASET.YOUTUBEBB.ANNO = 'train_dataset/yt_bb/train.json'
# __C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
# __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat
#
# __C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = 'train_dataset/coco/crop511'         # COCO dataset path
# __C.DATASET.COCO.ANNO = 'train_dataset/coco/train2017.json'
# __C.DATASET.COCO.FRAME_RANGE = 1
# __C.DATASET.COCO.NUM_USE = -1
#
# __C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = 'train_dataset/det/crop511'           # DET dataset path
# __C.DATASET.DET.ANNO = 'train_dataset/det/train.json'
# __C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = -1
#
# __C.DATASET.GOT = CN()
# __C.DATASET.GOT.ROOT = 'train_dataset/got10k/crop511'         # GOT dataset path
# __C.DATASET.GOT.ANNO = 'train_dataset/got10k/train.json'
# __C.DATASET.GOT.FRAME_RANGE = 50
# __C.DATASET.GOT.NUM_USE = 100000
#
# __C.DATASET.LaSOT = CN()
# __C.DATASET.LaSOT.ROOT = 'train_dataset/lasot/crop511'         # LaSOT dataset path
# __C.DATASET.LaSOT.ANNO = 'train_dataset/lasot/train.json'
# __C.DATASET.LaSOT.FRAME_RANGE = 100
# __C.DATASET.LaSOT.NUM_USE = 100000
#
# __C.DATASET.VIDEOS_PER_EPOCH = 600000 #600000

############################ORIGINAL#############################
__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '/data0/ILSVRC2015_VID/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  #repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = '/data1/yt_bb'
# __C.DATASET.YOUTUBEBB.ROOT = '/data0/youtubebb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000  # use all not repeat
# __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '/data0/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = 100000
# __C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = '/data0/DET/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 200000
# __C.DATASET.DET.NUM_USE = -1

__C.DATASET.MOT = CN()
__C.DATASET.MOT.ROOT = '/data0/MOT/crop511/train'
__C.DATASET.MOT.ANNO = 'training_dataset/mot/train.json'
__C.DATASET.MOT.FRAME_RANGE = 100
__C.DATASET.MOT.NUM_USE = -1

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = '/data1/got10k_training/crop511'
__C.DATASET.GOT.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 100
__C.DATASET.GOT.NUM_USE = 200000 # 100000
# __C.DATASET.GOT.FRAME_RANGE = 50
# __C.DATASET.GOT.NUM_USE = 100000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = '/data0/LaSOT/crop511'
__C.DATASET.LaSOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 200000 # 100000
# __C.DATASET.LaSOT.FRAME_RANGE = 100
# __C.DATASET.LaSOT.NUM_USE = 100000

__C.DATASET.TrackingNet = CN()
__C.DATASET.TrackingNet.ROOT = '/data1/TrackingNet/crop511'
__C.DATASET.TrackingNet.ANNO = 'training_dataset/trackingnet/train.json'
__C.DATASET.TrackingNet.FRAME_RANGE = 100
__C.DATASET.TrackingNet.NUM_USE = 200000

# __C.DATASET.VIDEOS_PER_EPOCH = 800000
__C.DATASET.VIDEOS_PER_EPOCH = 600000

# ##########################MODIFY refer to Ocean###########################
# __C.DATASET.VID = CN()
# __C.DATASET.VID.ROOT = '/data0/ILSVRC2015_VID/crop511'
# __C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
# __C.DATASET.VID.FRAME_RANGE = 100
# __C.DATASET.VID.NUM_USE = 100000  #repeat until reach NUM_USE
#
# __C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = '/data1/yt_bb'
# # __C.DATASET.YOUTUBEBB.ROOT = '/data0/youtubebb/crop511'
# __C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
# __C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
# __C.DATASET.YOUTUBEBB.NUM_USE = 200000  # use all not repeat
# # __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat
#
# __C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = '/data0/coco/crop511'
# __C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
# __C.DATASET.COCO.FRAME_RANGE = 1
# __C.DATASET.COCO.NUM_USE = 50000 # 100000
# # __C.DATASET.COCO.NUM_USE = -1
#
# __C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = '/data0/DET/crop511'
# __C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
# __C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = 50000 # 200000
# # __C.DATASET.DET.NUM_USE = -1
#
# __C.DATASET.GOT = CN()
# __C.DATASET.GOT.ROOT = '/data1/got10k_training/crop511'
# __C.DATASET.GOT.ANNO = 'training_dataset/got10k/train.json'
# __C.DATASET.GOT.FRAME_RANGE = 100
# __C.DATASET.GOT.NUM_USE = 200000
# # __C.DATASET.GOT.FRAME_RANGE = 50
# # __C.DATASET.GOT.NUM_USE = 100000
#
# # __C.DATASET.LaSOT = CN()
# # __C.DATASET.LaSOT.ROOT = '/data0/LaSOT/crop511'
# # __C.DATASET.LaSOT.ANNO = 'training_dataset/lasot/train.json'
# # __C.DATASET.LaSOT.FRAME_RANGE = 100
# # __C.DATASET.LaSOT.NUM_USE = 50000
# # __C.DATASET.LaSOT.FRAME_RANGE = 100
# # __C.DATASET.LaSOT.NUM_USE = 200000
#
# __C.DATASET.VIDEOS_PER_EPOCH = 600000

#
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

# SiamGAT
__C.TRAIN.ATTENTION = False

__C.TRAIN.ATTENTION_NUM = 4

# Siamcar resnet50
__C.TRAIN.DOWN = False

# # googlenet
# __C.TRAIN.CHANNEL = False

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.OFFSET = 45

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB50 = [0.15, 0.1, 0.4]

__C.HP_SEARCH.GOT10K = [0.6, 0.04, 0.1]

__C.HP_SEARCH.UAV123 = [0.39, 0.04, 0.37]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.4]
