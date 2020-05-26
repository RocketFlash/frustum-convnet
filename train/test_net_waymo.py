from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0, 7"

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
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg

from utils.training_states import TrainingStates
from utils.utils import get_accuracy, AverageMeter, import_from_file, get_logger
from utils.plot_utils import render_sample_3d_interactive

from datasets.dataset_info import DATASET_INFO

from ops.pybind11.rbbox_iou import bev_nms_np
from ops.pybind11.rbbox_iou import cube_nms_np
import kitti.kitti_util as utils
from waymo.waymo_utils import get_sample_names
from inference import predict
from eval.evaluate import calculate_3d_mAP, get_lyft_format_data

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See configs/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def set_random_seed(seed=3):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_nms(det_results, threshold=cfg.TEST.THRESH):
    nms_results = {}
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = np.array(det_results[idx][class_type], dtype=np.float32)
            if len(dets) > 1:
                dets_for_nms = dets[:, 4:][:, [0, 1, 2, 5, 4, 3, 6, 7]]
                keep = cube_nms_np(dets_for_nms, threshold)
                dets_keep = dets[keep]
            else:
                dets_keep = dets
            if idx not in nms_results:
                nms_results[idx] = {}
            nms_results[idx][class_type] = dets_keep
    return nms_results



def test(model, test_dataset, test_loader, output_filename, result_dir=None):

    load_batch_size = test_loader.batch_size
    num_batches = len(test_loader)

    model.eval()

    fw_time_meter = AverageMeter()

    det_results = {}
    gt_results = {}

    for i, (data_dicts, datas) in enumerate(test_loader):
        torch.cuda.synchronize()
        tic = time.time()
        
        if data_dicts is None:
            continue
        predictions = predict(model, data_dicts)

        torch.cuda.synchronize()
        fw_time_meter.update((time.time() - tic))
        print('%d/%d %.3f' % (i, num_batches, fw_time_meter.val))

        # datas = test_dataset.get_frustum_data(i)
        # print(f'Datas len: {len(datas)}')
        # print(f'Predictions len: {len(predictions)}')

        for data, pred in zip(datas, predictions):
                
            if data is None:
                continue

            data_idx = data['idx_i']
            class_type = data['type_i']
            box2d = data['box2d_i']
            box3d = data['box3d_i']
            box3d_sizes = data['box3d_size_i']
            ry_gt = data['heading_i']
            box_pos_gt = data['obj_pos_i']

            if data_idx not in det_results:
                det_results[data_idx] = {}
                gt_results[data_idx] = {}

            if class_type not in det_results[data_idx]:
                det_results[data_idx][class_type] = []
                gt_results[data_idx][class_type] = []

            x1, y1, x2, y2 = box2d
            l_gt,w_gt,h_gt = box3d_sizes
            tx_gt, ty_gt, tz_gt = box_pos_gt

            output_gt = [x1, y1, x2,  y2, tx_gt, ty_gt, tz_gt, h_gt, w_gt, l_gt, ry_gt]
            gt_results[data_idx][class_type].append(output_gt)

            # print('****************')
            # print(tx_gt, ty_gt, tz_gt, h_gt, w_gt, l_gt, ry_gt)
            # print('================')
            for n in range(len(pred)):  
                h, w, l, tx, ty, tz, ry, score = pred[n]
                output = [x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry, score]
                # print(tx, ty, tz, h, w, l, ry)
                # output = [x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry,1]
                # print(score)
                det_results[data_idx][class_type].append(output)
            # print('****************')

    num_images = len(det_results)

    logging.info('Average time:')
    logging.info('batch:%0.3f' % fw_time_meter.avg)
    logging.info('avg_per_object:%0.3f' % (fw_time_meter.avg / load_batch_size))
    logging.info('avg_per_image:%.3f' % (fw_time_meter.avg * len(test_loader) / num_images))

    return gt_results, det_results


if __name__ == '__main__':

    set_random_seed()
    args = parse_args()

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    assert_and_infer_cfg()

    SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.SAVE_SUB_DIR)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # set logger
    cfg_name = os.path.basename(args.cfg_file).split('.')[0]
    log_file = '{}_{}_val.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))

    logger = get_logger(os.path.join(SAVE_DIR, log_file))
    logger.info('config:\n {}'.format(pprint.pformat(cfg)))

    model_def = import_from_file(cfg.MODEL.FILE)
    model_def = model_def.PointNetDet

    dataset_def = import_from_file(cfg.DATA.FILE)
    collate_fn = dataset_def.collate_fn
    dataset_def = dataset_def.ProviderDataset


    test_dataset_paths = cfg.DATA.TEST_DATASET_PATHS
    test_datasets, test_dataset_idxs = get_sample_names(test_dataset_paths[0])
    X_test = test_dataset_idxs

    logging.info(f'Number of test samples: {len(X_test)}')

    logging.info('Load test dataset')

    test_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        data_idxs_list=X_test,
        datasets = test_datasets,
        lidar_points_threshold=5,
        classes=cfg.MODEL.CLASSES,
        one_hot=True,
        random_flip=False,
        random_shift=False,
        filter_objects=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)

    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else 4

    dataset_name = cfg.DATA.DATASET_NAME
    assert dataset_name in DATASET_INFO
    datset_category_info = DATASET_INFO[dataset_name]
    NUM_VEC = len(datset_category_info.CLASSES) # rgb category as extra feature vector
    NUM_CLASSES = len(cfg.MODEL.CLASSES)

    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)
    model = model.cuda()

    if os.path.isfile(cfg.TEST.WEIGHTS):
        checkpoint = torch.load(cfg.TEST.WEIGHTS)

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TEST.WEIGHTS, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("=> loaded checkpoint '{}')".format(cfg.TEST.WEIGHTS))
    else:
        logging.error("=> no checkpoint found at '{}'".format(cfg.TEST.WEIGHTS))
        assert False

    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model, device_ids = [1, 0])

    save_file_name = os.path.join(SAVE_DIR, 'detection.pkl')
    result_folder = os.path.join(SAVE_DIR, 'result')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    gt_results, det_results = test(model, test_dataset, test_loader, save_file_name)

    # Write detection results for KITTI evaluation
    print('Detection results')
    print(f'detections len: {len(det_results)}')
    print(f'gt len: {len(gt_results)}')
    if cfg.TEST.METHOD == 'nms':
        det_results = make_nms(det_results)

    gt_results_lyft, det_results_lyft = get_lyft_format_data(gt_results, det_results)
    print(len(gt_results_lyft))
    print(len(det_results_lyft))
    calculate_3d_mAP(gt_results_lyft, det_results_lyft, iou_threshold=0.5)
    # evaluate_cuda_wrapper(gt_results, det_results)