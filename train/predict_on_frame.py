from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,7"

import sys
import math
import shutil
import time
import argparse

import pprint
import random as pyrandom
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import pickle

from train.inference import load_model, predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def get_frustum_input(self, data):
    box2d_i = data['box2d_i']
    input_i = data['input_i']
    type_i = data['type_i']
    frustum_angle_i = data['frustum_angle_i']
    calib_i = data['calib_i']

    rotate_to_center = cfg.DATA.RTC
    with_extra_feat = cfg.DATA.WITH_EXTRA_FEAT

    ''' Get index-th element from the picked file dataset. '''
    # ------------------------------ INPUTS ----------------------------
    rot_angle = self.get_center_view_rot_angle(frustum_angle_i )

    cls_type = type_i
    assert cls_type in WaymoCategory.CLASSES, cls_type
    size_class = WaymoCategory.CLASSES.index(cls_type)

    # Compute one hot vector
    if self.one_hot:
        one_hot_vec = np.zeros((3))
        one_hot_vec[size_class] = 1

    # Get point cloud
    if rotate_to_center:
        point_set = self.get_center_view_point_set(input_i, frustum_angle_i)
    else:
        point_set = input_i 

    if not with_extra_feat:
        point_set = point_set[:, :3]

    # Resample
    if self.npoints > 0:
        choice = np.random.choice(point_set.shape[0], self.npoints, point_set.shape[0] < self.npoints)
    else:
        choice = np.random.permutation(len(point_set.shape[0]))

    point_set = point_set[choice, :]

    box = box2d_i
    P = calib_i.P

    ref1, ref2, ref3, ref4 = self.generate_ref(box, P)

    if rotate_to_center:
        ref1 = self.get_center_view(ref1, frustum_angle_i)
        ref2 = self.get_center_view(ref2, frustum_angle_i)
        ref3 = self.get_center_view(ref3, frustum_angle_i)
        ref4 = self.get_center_view(ref4, frustum_angle_i)


    data_inputs = {
        'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
        'rot_angle': torch.FloatTensor([rot_angle]),
        'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
        'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
        'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
        'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),
    }

    if not rotate_to_center:
        data_inputs.update({'rot_angle': torch.zeros(1)})

    if self.one_hot:
        data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})
    
    return data_inputs



def main():
    TFRECORD_FILE_PATH = ''
    FRAME_NUMBER = ''
    CONFIG_FILE = ''

    dataset = tf.data.TFRecordDataset(file_name, compression_type='')

    frame = open_dataset.Frame()

    for idx, data_i in enumerate(self.datasets[dataset]):
        if idx == FRAME_NUMBER:     
            frame.ParseFromString(bytearray(data_i.numpy()))

    images = wutils.get_images(frame)
    calibs = wutils.get_calibs(frame)
    pc = wutils.get_lidar(frame)
    image_shapes = [img.shape for img in images]
    data = extract_frustum_data_inference(bbox, pc, img, calib, type_whitelist=classes)
    get_frustum_input(data)
    model = load_model(cfg_file, testing=True)
    predictions = predict(model, data_dicts)


if __name__ == '__main__':
    main()