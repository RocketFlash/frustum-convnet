from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append('frustum-convnet')

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
from waymo.prepare_data_waymo import extract_frustum_data_inference
import tensorflow as tf

import waymo.waymo_utils as wutils
from waymo.filter_data import filter_boxes, filter_boxes_kitti
from datasets.dataset_info import WaymoCategory

from waymo_open_dataset import dataset_pb2 as open_dataset
from datasets.data_utils import rotate_pc_along_y, project_image_to_rect, compute_box_3d, extract_pc_in_box3d, roty


def generate_ref(box, P):
    s1, s2, s3, s4 = (0.1, 0.2, 0.4, 0.8)

    z1 = np.arange(0, 70, s1) + s1 / 2.
    z2 = np.arange(0, 70, s2) + s2 / 2.
    z3 = np.arange(0, 70, s3) + s3 / 2.
    z4 = np.arange(0, 70, s4) + s4 / 2.

    cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,

    xyz1 = np.zeros((len(z1), 3))
    xyz1[:, 0] = cx
    xyz1[:, 1] = cy
    xyz1[:, 2] = z1
    xyz1_rect = project_image_to_rect(xyz1, P)

    xyz2 = np.zeros((len(z2), 3))
    xyz2[:, 0] = cx
    xyz2[:, 1] = cy
    xyz2[:, 2] = z2
    xyz2_rect = project_image_to_rect(xyz2, P)

    xyz3 = np.zeros((len(z3), 3))
    xyz3[:, 0] = cx
    xyz3[:, 1] = cy
    xyz3[:, 2] = z3
    xyz3_rect = project_image_to_rect(xyz3, P)

    xyz4 = np.zeros((len(z4), 3))
    xyz4[:, 0] = cx
    xyz4[:, 1] = cy
    xyz4[:, 2] = z4
    xyz4_rect = project_image_to_rect(xyz4, P)

    return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect


def get_center_view_rot_angle(frustum_angle):
    ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
    can be directly used to adjust GT heading angle '''
    return np.pi / 2.0 + frustum_angle


def get_box3d_center(box3d):
    ''' Get the center (XYZ) of 3D bounding box. '''
    box3d_center = (box3d[0, :] + box3d[6, :]) / 2.0
    return box3d_center


def get_center_view_box3d_center(box3d, frustum_angle):
    ''' Frustum rotation of 3D bounding box center. '''
    box3d_center = (box3d[0, :] + box3d[6, :]) / 2.0
    return rotate_pc_along_y(np.expand_dims(box3d_center, 0), get_center_view_rot_angle(frustum_angle)).squeeze()


def get_center_view_box3d(box3d, frustum_angle):
    ''' Frustum rotation of 3D bounding box corners. '''
    box3d = box3d
    box3d_center_view = np.copy(box3d)
    return rotate_pc_along_y(box3d_center_view, get_center_view_rot_angle(frustum_angle))


def get_center_view_point_set(input_i, frustum_angle):
    ''' Frustum rotation of point clouds.
    NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    '''
    # Use np.copy to avoid corrupting original data
    point_set = np.copy(input_i)
    return rotate_pc_along_y(point_set, get_center_view_rot_angle(frustum_angle))


def get_center_view(point_set, frustum_angle):
    ''' Frustum rotation of point clouds.
    NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    '''
    # Use np.copy to avoid corrupting original data
    point_set = np.copy(point_set)
    return rotate_pc_along_y(point_set, get_center_view_rot_angle(frustum_angle))


def get_frustum_input(data,  one_hot=True, npoints=1024, min_n_points = 5):
    box2d_i = data['box2d_i']
    input_i = data['input_i']
    type_i = data['type_i']
    frustum_angle_i = data['frustum_angle_i']
    calib_i = data['calib_i']

    rotate_to_center = True
    with_extra_feat = False

    ''' Get index-th element from the picked file dataset. '''
    # ------------------------------ INPUTS ----------------------------
    rot_angle = get_center_view_rot_angle(frustum_angle_i )

    cls_type = type_i
    assert cls_type in WaymoCategory.CLASSES, cls_type
    size_class = WaymoCategory.CLASSES.index(cls_type)

    # Compute one hot vector
    if one_hot:
        one_hot_vec = np.zeros((3))
        one_hot_vec[size_class] = 1

    # Get point cloud
    if rotate_to_center:
        point_set = get_center_view_point_set(input_i, frustum_angle_i)
    else:
        point_set = input_i 

    if not with_extra_feat:
        point_set = point_set[:, :3]

    if point_set.shape[0]<= min_n_points:
        # print(f'No enougth number of points {point_set.shape[0]}')
        return None
    # Resample
    if npoints > 0:
        choice = np.random.choice(point_set.shape[0], npoints, point_set.shape[0] < npoints)
    else:
        choice = np.random.permutation(len(point_set.shape[0]))

    point_set = point_set[choice, :]

    box = box2d_i
    P = calib_i.P

    ref1, ref2, ref3, ref4 = generate_ref(box, P)

    if rotate_to_center:
        ref1 = get_center_view(ref1, frustum_angle_i)
        ref2 = get_center_view(ref2, frustum_angle_i)
        ref3 = get_center_view(ref3, frustum_angle_i)
        ref4 = get_center_view(ref4, frustum_angle_i)


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

    if one_hot:
        data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})
    
    return data_inputs


def predict_frame(model, file_name, frame_number, bboxes_2d):
    dataset = tf.data.TFRecordDataset(file_name, compression_type='')
    frame = open_dataset.Frame()

    for idx, data_i in enumerate(dataset):
        if idx == frame_number:     
            frame.ParseFromString(bytearray(data_i.numpy()))

    images = wutils.get_images(frame)
    calibs = wutils.get_calibs(frame)
    pc = wutils.get_lidar(frame)

    predictions_all = [None] * 5
    datas_all = [None] * 5

    for cam_idx, (img, calib, bboxes) in enumerate(zip(images, calibs, bboxes_2d)):
        # print('##############################################')
        # print(cam_idx,  bboxes)
        data_dicts = []
        datas = []
        for bbox in bboxes:
            data_dict = None
            data = extract_frustum_data_inference(bbox, pc, img, calib, type_whitelist=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'])
            if data is not None:
                data_dict = get_frustum_input(data)
            if data_dict is not None:
                # print(cam_idx)
                data_dicts.append(data_dict)
                datas.append(data)
        datas_all[cam_idx] = datas

        total_dict = {}
        for dict_i in data_dicts:
            for k, v in dict_i.items():
                if k not in total_dict:
                    total_dict[k] = v.unsqueeze(0)
                else:
                    total_dict[k] = torch.cat((total_dict[k], v.unsqueeze(0)), 0) 
        if total_dict:
            # print(cam_idx)
            predictions = predict(model, total_dict)
            predictions_all[cam_idx] = predictions

    # print(f'Number of objects on camera :  {sum([len(bboxes) for bboxes in bboxes_2d])}')
    # print(f'Number of objects to frustum :  {len(datas)}')
    
    return predictions_all, datas_all
