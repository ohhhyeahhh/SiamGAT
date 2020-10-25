# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # # build adjust layer
        # if cfg.ADJUST.ADJUST:
        #     self.neck = get_neck(cfg.ADJUST.TYPE,
        #                          **cfg.ADJUST.KWARGS)

        # build rpn head
        # self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
        #                              **cfg.RPN.KWARGS)
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)
        # self.down = nn.ConvTranspose2d(cfg.HEAD.INCHANNEL * 3, cfg.HEAD.INCHANNEL, 1, 1)
        # self.down = nn.Conv2d(256 * 2, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        # # get features map
        # for idx, (z, x) in enumerate(zip(self.zf, xf)):
        #     attention = getattr(self, 'attention' + str(idx))
        #     if idx == 0:
        #         features = attention(z, x)
        #     else:
        #         feature = attention(z, x)
        #         features += feature
        #         # features = torch.cat([feature, features], 1)
        #
        # features = self.down(features)

        # features = self.gen_attention(xf[0], self.zf[0])
        # for i in range(len(xf)-1):
        #     xf_new = self.gen_attention(xf[i+1], self.zf[i+1])
        #     features = torch.cat([features,xf_new],1)
        # features = self.down(features)

        # features = self.xcorr_depthwise(xf[0], self.zf[0])
        # for i in range(len(xf)-1):
        #     features_new = self.xcorr_depthwise(xf[i+1], self.zf[i+1])
        #     features = torch.cat([features,features_new],1)
        # features = self.down(features)

        features = self.xcorr_depthwise(xf, self.zf)
        cls, loc, cen = self.car_head(features)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        # targets = data['target']
        neg = data['neg'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        # label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        # if cfg.ADJUST.ADJUST:
        #     zf = self.neck(zf)
        #     xf = self.neck(xf)

        features = self.xcorr_depthwise(xf, zf)

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.INSTANCE_SIZE)
        # if cls_loss is cross_entropy_loss, add this
        cls = self.log_softmax(cls)

        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss

        return outputs
