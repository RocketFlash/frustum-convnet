''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


import kitti.kitti_util as utils
from kitti.kitti_object import kitti_object
from kitti.draw_util import get_lidar_in_image_fov

from ops.pybind11.rbbox_iou import bbox_overlaps_2d
from tqdm import tqdm


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def random_shift_box2d(box2d, img_height, img_width, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    assert xmin < xmax and ymin < ymax

    while True:
        cx2 = cx + w * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        new_box2d = np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])

        new_box2d[[0, 2]] = np.clip(new_box2d[[0, 2]], 0, img_width - 1)
        new_box2d[[1, 3]] = np.clip(new_box2d[[1, 3]], 0, img_height - 1)

        if new_box2d[0] < new_box2d[2] and new_box2d[1] < new_box2d[3]:
            return new_box2d


def extract_boxes(objects, type_whitelist):
    boxes_2d = []
    boxes_3d = []

    filter_objects = []

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        if obj.type not in type_whitelist:
            continue

        boxes_2d += [obj.box2d]
        boxes_3d += [np.array([obj.t[0], obj.t[1], obj.t[2], obj.l, obj.w, obj.h, obj.ry])]
        filter_objects += [obj]

    if len(boxes_3d) != 0:
        boxes_3d = np.stack(boxes_3d, 0)
        boxes_2d = np.stack(boxes_2d, 0)

    return filter_objects, boxes_2d, boxes_3d


def extract_pc(data_name, dataset):
    calib = dataset.get_calibration(data_name)  # 3 by 4 matrix
    pc_velo = dataset.get_lidar(data_name)
    return pc_velo


def extract_image(data_name, dataset):
    img = dataset.get_image(data_name)
    return img


def extract_frustum_data(object_i, pc, img, calib, idx_i=0, perturb_box2d=False, augmentX=1, type_whitelist=['Car'], ):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        data_names_lists: names list
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''

    if object_i.type not in type_whitelist:
        return None
        
    data = {}

    pc_velo = pc
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_rect[:, 3] = pc_velo[:, 3]

    if img is None:
        print('SKIP IMAGE!')
        return None
    img_height, img_width, img_channel = img.shape
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)

    # 2D BOX: Get pts rect backprojected
    box2d = object_i.box2d
    for _ in range(augmentX):
        # Augment data by box2d perturbation
        if perturb_box2d:
            xmin, ymin, xmax, ymax = random_shift_box2d(box2d, img_height, img_width, 0.1)
        else:
            xmin, ymin, xmax, ymax = box2d
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                    (pc_image_coord[:, 0] >= xmin) & \
                    (pc_image_coord[:, 1] < ymax) & \
                    (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]

        pc_box_image_coord = pc_image_coord[box_fov_inds]

        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])
        # 3D BOX: Get pts velo in 3d box
        obj = object_i
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
        label = np.zeros((pc_in_box_fov.shape[0]))
        label[inds] = 1

        # Get 3D BOX heading
        heading_angle = obj.ry
        # Get 3D BOX size
        box3d_size = np.array([obj.l, obj.w, obj.h])
        # Object position
        obj_pos_i = obj.t

        # Reject too far away object or object without points
        if  (box2d[3] - box2d[1]) < 25 or np.sum(label) == 0:
            # print(box2d[3] - box2d[1], np.sum(label))
            return None

        box2d_i=np.array([xmin, ymin, xmax, ymax])
        box3d_i=box3d_pts_3d
        input_i=pc_in_box_fov.astype(np.float32, copy=False)
        label_i=label
        type_i=object_i.type
        heading_i=heading_angle
        box3d_size_i=box3d_size
        frustum_angle_i=frustum_angle

        gt_box2d_i=box2d
        calib_i=calib
        data = {'idx_i' : idx_i,
                'box2d_i' : box2d_i, 
                'box3d_i': box3d_i,
                'obj_pos_i': obj_pos_i ,
                'input_i':input_i, 
                'label_i':label_i,
                'type_i':type_i, 
                'heading_i':heading_i, 
                'box3d_size_i':box3d_size_i, 
                'frustum_angle_i':frustum_angle_i,
                'gt_box2d_i':gt_box2d_i, 
                'calib_i':calib_i}
    return data
  

def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare':
                continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def read_det_pkl_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    with open(det_filename, 'r') as fn:
        results = pickle.load(fn)

    id_list = results['id_list']
    type_list = results['type_list']
    box2d_list = results['box2d_list']
    prob_list = results['prob_list']

    return id_list, type_list, box2d_list, prob_list


