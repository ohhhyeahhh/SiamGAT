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
from pysot.models.head.car_head import CARHead_CT as head
from ..utils.location_grid import compute_locations
from pysot.models.neck import get_neck


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel, fore_groups=1, back_groups=1):
        super(Graph_Attention_Union, self).__init__()

        self.fore_groups = fore_groups
        self.back_groups = back_groups
        for i in range(fore_groups):
            # foreground nodes linear transformation
            self.add_module('fore_support'+str(i),nn.Conv2d(in_channel, in_channel, 1, 1, bias=False))
            # search nodes linear transformation
            self.add_module('fore_query'+str(i),nn.Conv2d(in_channel, in_channel, 1, 1, bias=False))
            # foreground transformation for message passing
            self.add_module('fore_g'+str(i),nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),)
            )
        for i in range(back_groups):
            # background nodes linear transformation
            self.add_module('back_support'+str(i),nn.Conv2d(in_channel, in_channel, 1, 1, bias=False))
            # search nodes linear transformation
            self.add_module('back_query'+str(i),nn.Conv2d(in_channel, in_channel, 1, 1, bias=False))
            # background transformation for message passing
            self.add_module('back_g'+str(i),nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),)
            )

        # search transformation for message passing
        self.xf_g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*(fore_groups+back_groups+1), out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf, zf_mask):
        zf_fore = zf * zf_mask
        zf_back = zf * (1 - zf_mask)

        similars = []
        ebds = []
        for idx in range(self.fore_groups):
            query_func = getattr(self, 'fore_query'+str(idx))
            support_func = getattr(self, 'fore_support'+str(idx))
            g_func = getattr(self, 'fore_g'+str(idx))
            similar, ebd = self.calculate(support_func(zf_fore), query_func(xf), g_func(zf_fore))
            similars.append(similar)
            ebds.append(ebd)

        for idx in range(self.back_groups):
            query_func = getattr(self, 'back_query'+str(idx))
            support_func = getattr(self, 'back_support'+str(idx))
            g_func = getattr(self, 'back_g'+str(idx))
            similar, ebd = self.calculate(support_func(zf_back), query_func(xf), g_func(zf_back))
            similars.append(similar)
            ebds.append(ebd)
        ebds = torch.cat(ebds, dim=1)

        # aggregated feature
        output = torch.cat([ebds, self.xf_g(xf)], 1)
        output = self.fi(output)
        return output

    def calculate(self, zf, xf, zf_g):
        xf = F.normalize(xf, dim=1)
        zf = F.normalize(zf, dim=1)
        zf_flatten = zf.flatten(2)
        xf_flatten = xf.flatten(2)
        zf_g_flatten = zf_g.flatten(2)
        similar = torch.einsum("bcn,bcm->bnm", xf_flatten, zf_flatten)
        bs, c, xw, xh = xf.shape
        embedding = torch.einsum("bcm, bnm->bcn", zf_g_flatten, similar)
        embedding = embedding.reshape(bs, c, xw, xh)
        return similar, embedding


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build response map
        self.attention = Graph_Attention_Union(cfg.TRAIN.CHANNEL_NUM, cfg.TRAIN.CHANNEL_NUM, fore_groups=1, back_groups=1)

        # build car head
        self.car_head = head(cfg)

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z, mask):
        zf = self.backbone(z)
        self.zf_mask = F.interpolate(mask, size=zf.shape[-1], mode='bilinear')

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)

        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.attention(self.zf, xf, self.zf_mask)

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
        neg = data['neg'].cuda()
        mask = data['mask'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        zf_mask = F.interpolate(mask, size=zf.shape[-1], mode='bilinear')
        features = self.attention(zf, xf, zf_mask)

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
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
