''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''

import os
import sys
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
import kitti_util as utils


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'image_0')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne')
        self.label_dir = os.path.join(self.root_dir, 'label_filtered_0')
        self.num_samples = len([f for f in os.scandir(self.label_dir) if f.is_file()])

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '{}.png'.format(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, '{}.bin'.format(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '{}.txt'.format(idx))
        return utils.Calibration(calib_filename, cam_idx=0)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '{}.txt'.format(idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


class kitti_object_video(object):
    ''' Load data for KITTI videos '''

    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename)
                                     for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename)
                                       for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        # assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib

