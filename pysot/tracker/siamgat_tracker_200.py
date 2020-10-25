# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
import matplotlib.pyplot as plt
from pysot.utils.misc import bbox_clip
# from pysot.utils.show_map import show_map


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        self.cfg = cfg
        self.score_size = cfg.SCORE_SIZE
        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning)
        self.score_size_up = 193
        self.model = model
        self.model.eval()

    def _convert_cls(self, score):
        score = F.softmax(score[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        score += 0.05

        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        scale = cfg.TRACK.EXEMPLAR_SIZE / s_z
        # start from 1:
        # c = (cfg.TRACK.EXEMPLAR_SIZE + 1) / 2
        c = (cfg.TRACK.EXEMPLAR_SIZE - 1) / 2
        roi = torch.tensor([[c - bbox[2] * scale / 2, c - bbox[3] * scale / 2,
                             c + bbox[2] * scale / 2, c + bbox[3] * scale / 2]])

        self.model.template(z_crop, roi)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, bboxes, penalty_lk):
        bboxes_w = bboxes[0, :, :] + bboxes[2, :, :]
        bboxes_h = bboxes[1, :, :] + bboxes[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = 45
        # dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def corse_location(self, hp_cls_up, cen_up, scale_score, lrtbs):
        upsize = cfg.TRACK.SCORE_SIZE * cfg.TRACK.STRIDE
        # upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_cls_up.argmax(), hp_cls_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE-1)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE-1)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(cen_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        # mask[max_r_up_hp - l_region:max_r_up_hp + r_region + 1, max_c_up_hp - t_region:max_c_up_hp + b_region + 1] = 1
        cen_up = cen_up * mask
        return cen_up

    def getCenter(self, hp_cls_up, cen_up, scale_score, lrtbs):
        # corse location
        cen_up = self.corse_location(hp_cls_up, cen_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(cen_up.argmax(), cen_up.shape)
        disp = self.accurate_location(max_r_up, max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy, cen_up

    def track(self, img):
    # def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        # focal loss
        # cen = outputs['cen'].data.cpu().numpy().squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, cfg.TRACK.PENALTY_K)
        # penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_cls = penalty * cls
        p_score_1 = p_cls * cen

        if cfg.TRACK.hanming:
            hp_score_1 = p_score_1 * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
            # hp_score_1 = p_score_1 * (1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_cls = p_cls

        # test a new tracker
        hp_score_1_up = cv2.resize(hp_score_1, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_1_up = cv2.resize(p_score_1, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        # cen_up = cv2.resize(cen, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs, (1, 2, 0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / (cfg.TRACK.SCORE_SIZE-1)
        # scale_score = upsize / cfg.TRACK.SCORE_SIZE

        # get center
        max_r_up, max_c_up, new_cx, new_cy, cen_crop_up = self.getCenter(hp_score_1_up, p_score_1_up, scale_score, lrtbs)
        # cen_crop = cv2.resize(cen_crop_up, (self.cfg.SCORE_SIZE, self.cfg.SCORE_SIZE), interpolation=cv2.INTER_CUBIC)
        # if args.vis:
        self.show_cls_map(np.array([cls, cen, hp_score_1]), x_crop)
        # get w h
        ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2])
        ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3])

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * cfg.TRACK.LR
        # lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr * ave_w / self.scale_z + (1 - lr) * self.size[0]
        new_height = lr * ave_h / self.scale_z + (1 - lr) * self.size[1]


        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
        }

    def show_cls_map(self, score_map, image):
        score = score_map.reshape(-1, self.cfg.SCORE_SIZE, self.cfg.SCORE_SIZE)
        x_image = image.permute(0, 2, 3, 1)[0].data.cpu().numpy()
        imgs = x_image
        for i in range(score.shape[0]):
            cls_show = cv2.cvtColor(cv2.resize(score[i].astype(np.float32), (self.cfg.INSTANCE_SIZE, self.cfg.INSTANCE_SIZE), interpolation=cv2.INTER_CUBIC),
                                    cv2.COLOR_GRAY2RGB) * 255.0
            cls_show = cv2.applyColorMap(np.asarray(cls_show, dtype=np.uint8), cv2.COLORMAP_JET)
            imgs = np.hstack([imgs, cls_show])
        cv2.imshow('cls_map', imgs / 255.0)
        # cv2.waitKey(0)
        if 0xFF == ord(' ') & cv2.waitKey(1):
            cv2.waitKey(0)
        cv2.waitKey(1)

