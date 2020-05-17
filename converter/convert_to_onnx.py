#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import glob
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# from lyft_dataset_sdk.lyftdataset import LyftDataset
# from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
# from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, points_in_box, quaternion_yaw
# from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, get_average_precisions


# from src.data import BEVTestImageDataset, LyftTestDataset

# from src.utils import ARTIFACTS_FOLDER, CLASSES, CLASSES_CLS, convert_boxes_to_evals, visualize_predictions, create_transformation_matrix_to_voxel_space, transform_points, save_prediction, print_avp_differences
# from src.hypers import voxel_size, z_offset, bev_shape, box_scale
# from src.inference import open_preds, calc_detection_box
# from src.cam_detection_rauf import get_3d_object_from_camera, get_detection_model, get_class_of_box3D, get_classification_model, render_sample_3d_interactive

# from src.box_fitting_rauf import correct_box_dimensions, correct_box_dimensions_by_frustum


from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg
# from src.frustum_convnet.utils.utils import import_from_file

from pathlib import Path
import sys
import importlib
import shutil
from pyntcloud import PyntCloud
# from src.calib import Calib

from utils.utils import import_from_file


use_cam = True

merge_cfg_from_file('cfgs/det_sample_waymo.yaml')
assert_and_infer_cfg()
fr_weigths_path = 'pretrained_models/car_waymo/model_best.pth'
fr_model_def = import_from_file('models/det_base_onnx.py')
fr_model_def = fr_model_def.PointNetDet
input_channels = 3
NUM_VEC = 0
NUM_CLASSES = 2
fr_model = fr_model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)
fr_model = torch.nn.DataParallel(fr_model)
if os.path.isfile(fr_weigths_path):

    checkpoint = torch.load(fr_weigths_path)
    if 'state_dict' in checkpoint:
        fr_model.load_state_dict(checkpoint['state_dict'])
    else:
        fr_model.load_state_dict(checkpoint)
fr_model = fr_model.cuda()
fr_model.eval()

size_class = 0
one_hot_vec = np.zeros((3))
one_hot_vec[size_class] = 1

data_dicts = {
        'point_cloud': np.expand_dims(np.transpose(np.zeros((1000,4)), (1, 0)).astype(np.float32), axis=0),

        # 'center_ref1': np.expand_dims(np.transpose(ref1, (1, 0)).astype(np.float32), axis=0),
        # 'center_ref2': np.expand_dims(np.transpose(ref2, (1, 0)).astype(np.float32), axis=0),
        # 'center_ref3': np.expand_dims(np.transpose(ref3, (1, 0)).astype(np.float32), axis=0),
        # 'center_ref4': np.expand_dims(np.transpose(ref4, (1, 0)).astype(np.float32), axis=0),

        # 'rgb_prob': np.expand_dims(np.array(rgb_prob).astype(np.float32), axis=0),
        # 'rot_angle': np.expand_dims(np.array(rot_angle).astype(np.float32), axis=0),

        'one_hot': np.expand_dims(one_hot_vec.astype(np.float32), axis=0)
    }

data_dicts_var = {key: torch.from_numpy(value).cuda() for key, value in data_dicts.items()}

torch.cuda.synchronize()


point_set=np.zeros((1,1024,3))
valid_1=np.zeros((1,1,280,1))
valid_2=np.zeros((1,1,140,1))
valid_3=np.zeros((1,1,70,1))
valid_4=np.zeros((1,1,35,1))   
grouped_pc1=np.zeros((1,3,280,32))
grouped_pc2=np.zeros((1,3,140,32))
grouped_pc3=np.zeros((1,3,70,32))
grouped_pc4=np.zeros((1,3,35,32)) 
with torch.no_grad():

    inputvec1=[
        data_dicts_var.get('one_hot').unsqueeze(-1).expand(-1, -1, 280),
        data_dicts_var.get('one_hot').unsqueeze(-1).expand(-1, -1, 140),
        data_dicts_var.get('one_hot').unsqueeze(-1).expand(-1, -1, 70),
        data_dicts_var.get('one_hot').unsqueeze(-1).expand(-1, -1, 35),
        torch.from_numpy(point_set).float().cuda(), 
        torch.from_numpy(valid_1).float().cuda(), 
        torch.from_numpy(grouped_pc1).float().cuda(), 
        torch.from_numpy(valid_2).float().cuda(), 
        torch.from_numpy(grouped_pc2).float().cuda(), 
        torch.from_numpy(valid_3).float().cuda(), 
        torch.from_numpy(grouped_pc3).float().cuda(), 
        torch.from_numpy(valid_4).float().cuda(), 
        torch.from_numpy(grouped_pc4).float().cuda(),               
        
        ]

    torch.onnx.export(fr_model.module, inputvec1, "./frustum.onnx", 
                      export_params=True, do_constant_folding=False, 
                      opset_version=9, input_names=["one_hot_vec1","one_hot_vec2","one_hot_vec3","one_hot_vec4","point_cloud"], 
                      output_names=["score","outputs"])


# if use_cam:
#     cam_det_model = get_detection_model()

# pred_box3ds = []

# frame_name = '000008'

# with torch.no_grad():
    
#     image_path = dataset_root + 'images/' + frame_name + '.jpg'
#     pc_path = dataset_root + 'lidar/' + frame_name + '.bin'
#     calibration_file_path = dataset_root + 'calib/' + frame_name + '.txt'

#     image = cv2.imread(image_path)
#     pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4)).T
#     pc_i = LidarPointCloud(pc)
    

#     calib = Calib(calibration_file_path)
#     output_path = image_path.replace('image_2', 'output')

#     if use_cam:
#         pred_3dboxes_cam = []
#         im =  get_3d_object_from_camera(pc_i, image, calib, cam_det_model, pred_3dboxes_cam, fr_model)
    
#     pred_box3d_all = []

#     pred_box3d_evals_all = []

#     if use_cam:
#         pred_box3d_all.extend(pred_3dboxes_cam)


#     pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4)).T
#     pc_i = LidarPointCloud(pc)
#     pc_i_points = pc_i.points[:3,:]
#     startAll=time.time()
#     start_time = time.time()
#     if use_frustum:
#         new_pred_box3ds = correct_box_dimensions_by_frustum(pc_i_points, pred_box3d_all, fr_model, calib, use_det_frustum)
#     else:
#         new_pred_box3ds = pred_box3d_all
#     print("Correction took {0} ".format(time.time() - start_time))

#     new_pred_box3ds = pred_box3d_all
#     pred_box3d_evals_all = convert_boxes_to_evals(new_pred_box3ds)
#     pred_box3ds.extend(pred_box3d_evals_all)

#     render_sample_3d_interactive(pc_i, new_pred_box3ds, render_sample=False, render_gt=False)
    
 
