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
# from pysot.models.loss_car_sameregion import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head_nobn import CARHead
# from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        # merge different size nodes
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # Linear change
        xf_fi = self.query(xf)
        zf_theta = self.support(zf)

        # same Linear change
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate e
        shape_x = xf_fi.shape
        shape_z = zf_theta.shape

        zf_plat_fi = zf_theta.view(-1, shape_z[1], shape_z[2] * shape_z[2])
        zf_plat_g = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[2]).permute(0, 2, 1)
        xf_plat_fi = xf_fi.view(-1, shape_x[1], shape_x[2] * shape_x[2]).permute(0, 2, 1)

        similar = torch.matmul(xf_plat_fi, zf_plat_fi)
        similar = F.softmax(similar, dim=2)

        zf_add = torch.matmul(similar, zf_plat_g).permute(0, 2, 1)
        zf_add = zf_add.view(-1, shape_z[1], shape_x[2], shape_x[2])
        # return zf_add, xf_g

        # cat attention nodes with search nodes
        new_xf = torch.cat([zf_add, xf_g], 1)
        new_xf = self.fi(new_xf)
        return new_xf


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.attention = Graph_Attention_Union(256, 256)

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z, roi):
        zf = self.backbone(z, roi)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.attention(self.zf, xf)

        # features = self.xcorr_depthwise(xf,self.zf)
        # for i in range(len(xf)-1):
        #     features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
        #     features = torch.cat([features,features_new],1)
        # features = self.down(features)

        # cls, loc, cen = self.rpn_head(features)
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

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
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        features = self.attention(zf, xf)

        # features = self.xcorr_depthwise(xf, zf)
        # for i in range(len(xf)-1):
        #     features_new = self.xcorr_depthwise(xf[i+1], zf[i+1])
        #     features = torch.cat([features, features_new], 1)
        # features = self.down(features)

        # cls, loc, cen = self.rpn_head(features)
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.INSTANCE_SIZE)
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
