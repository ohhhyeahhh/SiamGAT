"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from pysot.models.focal_loss import FocalLoss

# from ..fcos_utils.sigmoid_focal_loss import SigmoidFocalLoss
from torch.autograd import Function
from torch.autograd.function import once_differentiable

INF = 100000000


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = FocalLoss()
        # self.cls_loss_func = SigmoidFocalLoss(
        #     cfg.TRAIN.LOSS_GAMMA,
        #     cfg.TRAIN.LOSS_ALPHA
        # )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox, neg):

        labels, reg_targets, num = self.compute_targets_for_locations(
            points, labels, gt_bbox, neg
        )

        return labels, reg_targets, num

    def compute_targets_for_locations(self, locations, labels, gt_bbox, neg):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2, -1)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

# ##########################################ignore labels################################################
        ignore_s1 = reg_targets_per_im[:, :, 0] > 0.2*((bboxes[:,2]-bboxes[:,0])/2).float()
        ignore_s2 = reg_targets_per_im[:, :, 2] > 0.2*((bboxes[:,2]-bboxes[:,0])/2).float()
        ignore_s3 = reg_targets_per_im[:, :, 1] > 0.2*((bboxes[:,3]-bboxes[:,1])/2).float()
        ignore_s4 = reg_targets_per_im[:, :, 3] > 0.2*((bboxes[:,3]-bboxes[:,1])/2).float()
        ignore_in_boxes = ignore_s1 * ignore_s2 * ignore_s3 * ignore_s4
        ignore_pos = np.where(ignore_in_boxes.cpu() == 1)
        labels[ignore_pos] = -1

        s1 = reg_targets_per_im[:, :, 0] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4

        pos = np.where(is_in_boxes.cpu() == 1)
        # num = pos[0].shape[0]
        # pos, pos_num = self.select(pos, self.cfg.TRAIN.POS_NUM)
        labels[pos] = 1
        labels = labels*(1-neg.long())
#####################################Ellipse Label####################################################
        # cx = (bboxes[:, 2].float() + bboxes[:, 0].float()) / 2
        # cy = (bboxes[:, 3].float() + bboxes[:, 1].float()) / 2
        # w = bboxes[:, 2].float() - bboxes[:, 0].float()
        # h = bboxes[:, 3].float() - bboxes[:, 1].float()
        # distance_e1 = self.ellipse(xs[:, None], ys[:, None], cx, cy, w / 2, h / 2)
        # distance_e2 = self.ellipse(xs[:, None], ys[:, None], cx, cy, w / 4, h / 4)
        # pos = np.where(distance_e2.cpu() <= 1)
        # ignore = np.where((distance_e1.cpu() <= 1) & (distance_e2.cpu() > 1))
        # labels[pos] = 1
        # labels[ignore] = -1
        # num = pos[0].shape[0]
        # labels = labels * (1 - neg.long())

        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous()

    def ellipse(self, x, y, cx, cy, w, h):
        distance = ((x - cx) / w)**2 + ((y - cy) / h)**2
        return distance

    def select(self, position, keep_num=16):
        # if len(position.shape) == 1:
        #     num = position.shape[0]
        # else:
        #     num = position[0].shape[0]
        num = position[0].shape[0]

        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        # return tuple(position[slt]), keep_num
        return tuple(p[slt] for p in position), keep_num

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets, neg):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets, neg)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        box_cls_flatten = box_cls.permute(0, 2, 3, 1).reshape(-1, self.cfg.TRAIN.NUM_CLASSES)
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        # cls_loss = self.cls_loss_func(box_cls, labels_flatten)
        # cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)
        N = box_cls[0].size(0)
        # focal loss config
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
