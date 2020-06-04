import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import logging
from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg
from utils.utils import get_accuracy, AverageMeter, import_from_file, get_logger
from datasets.provider_sample_waymo import from_prediction_to_label_format
from datasets.dataset_info import DATASET_INFO
from collections import OrderedDict

def predict(model, data_dicts, method='nms'):
    
    model.eval()

    point_clouds = data_dicts['point_cloud']
    rot_angles = data_dicts['rot_angle']
    # optional
    ref_centers = data_dicts.get('ref_center')
    rgb_probs = data_dicts.get('rgb_prob')

    # from ground truth box detection
    if rgb_probs is None:
        rgb_probs = torch.ones_like(rot_angles)

    # not belong to refinement stage
    if ref_centers is None:
        ref_centers = torch.zeros((point_clouds.shape[0], 3))
    
    batch_size = point_clouds.shape[0]
    rot_angles = rot_angles.view(-1)
    rgb_probs = rgb_probs.view(-1)

    if 'box3d_center' in data_dicts:
        data_dicts.pop('box3d_center')

    data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

    torch.cuda.synchronize()
    with torch.no_grad():
        outputs = model(data_dicts_var)

    cls_probs, center_preds, heading_preds, size_preds = outputs


    num_pred = cls_probs.shape[1]

    cls_probs = cls_probs.data.cpu().numpy()
    center_preds = center_preds.data.cpu().numpy()
    heading_preds = heading_preds.data.cpu().numpy()
    size_preds = size_preds.data.cpu().numpy()

    rgb_probs = rgb_probs.numpy()
    rot_angles = rot_angles.numpy()
    ref_centers = ref_centers.numpy()
    predictions = []
    for b in range(batch_size):

        if method == 'nms':
            fg_idx = (cls_probs[b, :, 0] < cls_probs[b, :, 1]).nonzero()[0]
            if fg_idx.size == 0:
                fg_idx = np.argmax(cls_probs[b, :, 1])
                fg_idx = np.array([fg_idx])
        else:
            fg_idx = np.argmax(cls_probs[b, :, 1])
            fg_idx = np.array([fg_idx])

        num_pred = len(fg_idx)

        single_centers = center_preds[b, fg_idx]
        single_headings = heading_preds[b, fg_idx]
        single_sizes = size_preds[b, fg_idx]
        single_scores = cls_probs[b, fg_idx, 1] + rgb_probs[b]
        
        rot_angle = rot_angles[b]
        ref_center = ref_centers[b]

        pred_i = []
        for n in range(num_pred):
            score = single_scores[n]
            h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
                single_centers[n], single_headings[n], single_sizes[n], rot_angle, ref_center)
            pred_i.append([h, w, l, tx, ty, tz, ry, score])
        predictions.append(pred_i)

    return predictions


def load_model(cfg_file, testing=False):
    merge_cfg_from_file(cfg_file)
    assert_and_infer_cfg()

    model_def = import_from_file(cfg.MODEL.FILE)
    model_def = model_def.PointNetDet

    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else 4
    dataset_name = cfg.DATA.DATASET_NAME
    assert dataset_name in DATASET_INFO
    datset_category_info = DATASET_INFO[dataset_name]
    NUM_VEC = len(datset_category_info.CLASSES) # rgb category as extra feature vector
    NUM_CLASSES = len(cfg.MODEL.CLASSES)

    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)
    model = model.cuda()

    if testing:
        if os.path.isfile(cfg.TEST.WEIGHTS):
            checkpoint = torch.load(cfg.TEST.WEIGHTS, map_location={'cuda:1': 'cuda:0'})
            if 'state_dict' in checkpoint:
                # print(checkpoint['state_dict'])
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                logging.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TEST.WEIGHTS, checkpoint['epoch']))
            else:
                model.load_state_dict(checkpoint)
                logging.info("=> loaded checkpoint '{}')".format(cfg.TEST.WEIGHTS))
        else:
            logging.error("=> no checkpoint found at '{}'".format(cfg.TEST.WEIGHTS))
            assert False

    return model