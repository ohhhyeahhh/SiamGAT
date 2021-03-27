#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

dataset_path = './data'

train_sets = ['TRAIN_0', 'TRAIN_1', 'TRAIN_2', 'TRAIN_3', 'TRAIN_4', 'TRAIN_5',
              'TRAIN_6', 'TRAIN_7', 'TRAIN_8', 'TRAIN_9', 'TRAIN_10', 'TRAIN_11']
test_sets = ['TEST']
d_sets = {'videos_train': train_sets}


def parse_and_sched(dl_dir='.'):
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            anno_path = os.path.join(dataset_path, dataset, 'anno')
            anno_files = os.listdir(anno_path)
            for video in anno_files:
                gt_path = os.path.join(anno_path, video)
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                for idx, gt_line in enumerate(groundtruth):
                    gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax
                    video_name = dataset + '/' + video.split('.')[0]
                    if video_name not in js:
                        js[video_name] = {}
                    if obj not in js[video_name]:
                        js[video_name][obj] = {}
                    js[video_name][obj][frame] = bbox
        if 'videos_test' == d_set:
            json.dump(js, open('test.json', 'w'), indent=4, sort_keys=True)
        else:
            json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)

        print(d_set+': All videos downloaded')


if __name__ == '__main__':
    parse_and_sched()
