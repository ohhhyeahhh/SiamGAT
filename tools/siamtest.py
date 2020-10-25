# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pysot.core.config import cfg
# from pysot.tracker.siamcar_tracker_upload import SiamCARTracker
# from pysot.tracker.siamgat_tracker import SiamCARTracker
from pysot.tracker.siamgat_tracker_UpMis import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
# from pysot.models.model_builder_single_gat_lasot import ModelBuilder
from pysot.models.model_builder_single_gat import ModelBuilder
# from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='LaSOT',
        help='datasets')#OTB50 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=False,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='snapshot/model_10_25_LaSOT.pth',
        help='snapshot of models to eval')

parser.add_argument('--config', type=str, default='../experiments/siamgat_googlenet/config.yaml',
        help='config file')


parser.add_argument('--lr', default=[0.4])
parser.add_argument('--penalty_k', default=[0.1, 0.05])
parser.add_argument('--window_lr', default=[0.4,0.3])
args = parser.parse_args()

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _check_and_occupation(result_path):
    if os.path.isfile(result_path):
        return True
    return False

def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = [0.35, 0.2, 0.55]

    # params = getattr(cfg.HP_SEARCH,args.dataset)
    # hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join('/data0', args.dataset)
    # dataset_root = os.path.join('/data1', args.dataset, 'test')
    # dataset_root = os.path.join(cur_dir,'../testing_dataset', args.dataset)
    if args.dataset == 'LaSOT':
        dataset_root = '/data0/LaSOT/LaSOT_test'
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    hp = {}
    for lr in args.lr:
        for pk in args.penalty_k:
            for w_lr in args.window_lr:
                # hp['penalty_k'] = pk
                # hp['lr'] = lr
                # hp['window_lr'] = w_lr
                cfg.TRACK.PENALTY_K = pk
                cfg.TRACK.WINDOW_INFLUENCE = w_lr
                cfg.TRACK.LR = lr

                model_name = args.snapshot.split('/')[-1].split('.')[-2] + '_' + str(lr) + '_' + str(pk) + '_' + str(w_lr)
                # model_name = args.snapshot.split('/')[-2] + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
                # OPE tracking
                for v_idx, video in enumerate(dataset):
                    if args.video != '':
                        # test one special video
                        if video.name != args.video:
                            continue
                    toc = 0
                    pred_bboxes = []
                    track_times = []
                    # result_path = os.path.join('results', args.dataset, model_name, '{}.txt'.format(video.name))
                    # if _check_and_occupation(result_path):
                    #     continue

                    # if v_idx < 3:
                    #     continue

                    for idx, (img, gt_bbox) in enumerate(video):
                        tic = cv2.getTickCount()

                        if idx == 0:
                            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                            tracker.init(img, gt_bbox_)
                            pred_bbox = gt_bbox_
                            pred_bboxes.append(pred_bbox)
                        else:
                            outputs = tracker.track(img)
                            # outputs = tracker.track(img, hp)
                            pred_bbox = outputs['bbox']
                            pred_bboxes.append(pred_bbox)
                        toc += cv2.getTickCount() - tic
                        track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                        if idx == 0:
                            cv2.destroyAllWindows()
                        if args.vis and idx > 0:
                            if not any(map(math.isnan,gt_bbox)):
                                gt_bbox = list(map(int, gt_bbox))
                                pred_bbox = list(map(int, pred_bbox))
                                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.imshow(video.name, img)
                                cv2.waitKey(1)
                    toc /= cv2.getTickFrequency()
                    # save results
                    model_path = os.path.join('/home/amax/PycharmProjects/results_of_ohhhyeahhh', args.dataset, model_name)
                    # model_path = os.path.join('results', args.dataset, model_name)
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                    print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                        v_idx+1, video.name, toc, idx / toc))
                # os.chdir(model_path)
                # save_file = '../%s' % dataset
                # shutil.make_archive(save_file, 'zip')
                # print('Records saved at', save_file + '.zip')


if __name__ == '__main__':
    main()
