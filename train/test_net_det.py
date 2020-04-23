from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

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

KITTI_EVAL_PATH = '../kitti-object-eval-python/'
sys.path.append(KITTI_EVAL_PATH)

import kitti_common as kitti
from eval import get_official_eval_result_my, get_coco_eval_result

from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg

from utils.training_states import TrainingStates
from utils.utils import get_accuracy, AverageMeter, import_from_file, get_logger
from utils.plot_utils import render_sample_3d_interactive

from datasets.provider_sample_my import from_prediction_to_label_format

from ops.pybind11.rbbox_iou import cube_nms_np
import kitti.kitti_util as utils


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


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


def write_detection_results(output_dir, gt_dirs, det_results):

    results = {}  # map from idx to list of strings, each string is a line (without \n)
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = det_results[idx][class_type]
            for i in range(len(dets)):
                output_str = class_type + " -1 -1 -10 "
                box2d = dets[i][:4]
                output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
                tx, ty, tz, h, w, l, ry = dets[i][4:-1]
                score = dets[i][-1]
                output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score)
                if idx not in results:
                    results[idx] = []
                results[idx].append(output_str)

    result_dir = os.path.join(output_dir, 'data')
    os.system('rm -rf %s' % (result_dir))
    os.mkdir(result_dir)

    for idx in results:
        pred_filename = os.path.join(result_dir, '{}.txt'.format(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()

    # Make sure for each frame (no matter if we have measurement for that frame),
    # there is a TXT file
    for gt_dir in gt_dirs:
        to_fill_filename_list = [f.name for f in os.scandir(gt_dir + 'label_2/') if f.is_file() and f.name.endswith('.txt')]
        fill_files(result_dir, to_fill_filename_list)


def write_detection_results_nms(output_dir, gt_dirs, det_results, threshold=cfg.TEST.THRESH):

    nms_results = {}
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = np.array(det_results[idx][class_type], dtype=np.float32)
            # scores = dets[:, -1]
            # keep = (scores > 0.001).nonzero()[0]
            # print(len(scores), len(keep))
            # dets = dets[keep]
            if len(dets) > 1:
                dets_for_nms = dets[:, 4:][:, [0, 1, 2, 5, 4, 3, 6, 7]]
                keep = cube_nms_np(dets_for_nms, threshold)
                # print(len(dets_for_nms), len(keep))
                dets_keep = dets[keep]
            else:
                dets_keep = dets
            if idx not in nms_results:
                nms_results[idx] = {}
            # if class_type not in nms_results[idx]:
            #     nms_results[idx][class_type] = []
            nms_results[idx][class_type] = dets_keep
    write_detection_results(output_dir, gt_dirs, nms_results)


def evaluate_py_wrapper(output_dir, gt_dirs, async_eval=False):
    # official evaluation
    for gt_dir in gt_dirs:
        gt_dir_labels = os.path.join(gt_dir, 'label_2/')
        command_line = './train/kitti_eval/evaluate_object_3d_offline %s %s' % (gt_dir_labels, output_dir)
        command_line += ' 2>&1 | tee -a  %s/log_test.txt' % (os.path.join(output_dir))
        print(command_line)
        if async_eval:
            subprocess.Popen(command_line, shell=True)
        else:
            if os.system(command_line) != 0:
                assert False


def evaluate_cuda_wrapper(result_dir, gt_dirs, classes_mapper, image_set='val', async_eval=False):
    # https://github.com/traveller59/kitti-object-eval-python
    # Sometime we can not get the same result as official evaluation for car BEV HARD detection (+-6%)
    if cfg.DATA.CAR_ONLY:
        classes_idx = '0'
    elif cfg.DATA.PEOPLE_ONLY:
        classes_idx = '1,2'
    else:
        classes_idx = '0,1,2'

    

    for gt_dir in gt_dirs:
        
        gt_dir_labels = os.path.join(gt_dir, 'label_2')
        gt_annos = kitti.get_label_annos(gt_dir_labels,classes_mapper)
        result_annos = kitti.get_label_annos(result_dir,classes_mapper)
        print(get_official_eval_result_my(gt_annos, result_annos, [0,1]))

        # CUDA_ID = 'CUDA_VISIBLE_DEVICES=3'
        # command_line = CUDA_ID + ('python ../kitti-object-eval-python/evaluate.py evaluate --label_path=%s ' +
        #                         '--result_path=%s --current_class=%s --coco=False') % (
        #     gt_dir_labels, result_dir, classes_idx)
        # command_line += ' 2>&1 | tee -a  %s/log_test_new.txt' % (os.path.join(result_dir, '..'))
        # print(command_line)
        # if async_eval:
        #     subprocess.Popen(command_line, shell=True)
        # else:
        #     if os.system(command_line) != 0:
        #         assert False


def test(model, test_dataset, test_loader, output_filename, classes_mapper, result_dir=None):

    load_batch_size = test_loader.batch_size
    num_batches = len(test_loader)

    model.eval()

    fw_time_meter = AverageMeter()

    det_results = {}

    for i, data_dicts in enumerate(test_loader):

        if i >= 1:
            break

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
        tic = time.time()
        with torch.no_grad():
            outputs = model(data_dicts_var)
            print(outputs[0].shape)

        cls_probs, center_preds, heading_preds, size_preds = outputs

        torch.cuda.synchronize()
        fw_time_meter.update((time.time() - tic))

        num_pred = cls_probs.shape[1]
        print('%d/%d %.3f' % (i, num_batches, fw_time_meter.val))

        cls_probs = cls_probs.data.cpu().numpy()
        center_preds = center_preds.data.cpu().numpy()
        heading_preds = heading_preds.data.cpu().numpy()
        size_preds = size_preds.data.cpu().numpy()

        rgb_probs = rgb_probs.numpy()
        rot_angles = rot_angles.numpy()
        ref_centers = ref_centers.numpy()

        for b in range(batch_size):

            if cfg.TEST.METHOD == 'nms':
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
            
            data = test_dataset.get_frustum_data(load_batch_size * i + b)
            pc_full = test_dataset.get_pointcloud(load_batch_size * i + b)
            img_full = test_dataset.get_image(load_batch_size * i + b)
            if data is None:
                continue
            data_idx = data['id_i']
            class_type = data['type_i']
            full_pc = data['input_i']
            box2d = data['box2d_i']
            rot_angle = rot_angles[b]
            ref_center = ref_centers[b]

            if data_idx not in det_results:
                det_results[data_idx] = {}

            if class_type not in det_results[data_idx]:
                det_results[data_idx][class_type] = []
            outputs_i = []
            for n in range(num_pred):
                x1, y1, x2, y2 = box2d
                score = single_scores[n]
                h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
                    single_centers[n], single_headings[n], single_sizes[n], rot_angle, ref_center)
                output = [x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry, score]
                outputs_i.append(output)
                det_results[data_idx][class_type].append(output)

    num_images = len(det_results)

    logging.info('Average time:')
    logging.info('batch:%0.3f' % fw_time_meter.avg)
    logging.info('avg_per_object:%0.3f' % (fw_time_meter.avg / load_batch_size))
    logging.info('avg_per_image:%.3f' % (fw_time_meter.avg * len(test_loader) / num_images))

    return det_results

def get_sample_names(dataset_paths, classes_mapper):
    total_dataset_names = []
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        image_names = [f.name.split('.')[0] for f in os.scandir(dataset_path + 'image_2/') if f.is_file() and f.name.endswith('.png')]
        lidar_names = [f.name.split('.')[0] for f in os.scandir(dataset_path + 'velodyne/') if f.is_file() and f.name.endswith('.bin')]
        calib_names = [f.name.split('.')[0] for f in os.scandir(dataset_path + 'calib/') if f.is_file() and f.name.endswith('.txt')]
        label_names = [f.name.split('.')[0] for f in os.scandir(dataset_path + 'label_2/') if f.is_file() and f.name.endswith('.txt')]
        sample_names = list(set(image_names).intersection(lidar_names))
        sample_names = list(set(sample_names).intersection(calib_names))
        sample_names = list(set(sample_names).intersection(label_names))
        
        if len(sample_names) > 0:
            for sample_name in sample_names:
                lines = [line.rstrip() for line in open(os.path.join(dataset_path,'label_2/',sample_name+'.txt'))]
                for obj_idx, line in enumerate(lines):
                    line_data = line.split(' ')
                    line_class = line_data[0]
                    if line_class in classes_mapper:
                        object_i = utils.Object3d(line)
                        box2d = object_i.box2d
                        if (box2d[3] - box2d[1]) < 25:
                            continue
                        total_dataset_names.append((dataset_idx, object_i, sample_name))
        else:
            print('Something wrong with data')
            sys.exit()
    return total_dataset_names

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

    classes_mapper = {'car':'car',
               'pedestrian' : 'pedestrian',
               'truck': 'car',
               'bicycle' : 'twowheels',
               'motorcycle' : 'twowheels',
               'VEHICLE':'car',
               'LARGE_VEHICLE':'car',
               'BICYCLE':'twowheels',
               'MOTORCYCLE' : 'twowheels',
               'PEDESTRIAN' : 'pedestrian',
               'MOPED' : 'twowheels',
               'Car' : 'car',
               'Van' : 'car',
               'Truck' : 'car',
               'Cyclist': 'twowheels',
               'Pedestrian' : 'pedestrian'
               }

    test_dataset_paths = cfg.DATA.TEST_DATASET_PATHS
    test_sample_names = get_sample_names(test_dataset_paths, classes_mapper)

    # overwritten_data_path = None
    # if cfg.OVER_WRITE_TEST_FILE and cfg.FROM_RGB_DET:
    #     overwritten_data_path = cfg.OVER_WRITE_TEST_FILE
    test_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        data_names_list=test_sample_names,
        classes_mapper=classes_mapper,
        dataset_paths = test_dataset_paths,
        classes=cfg.MODEL.CLASSES,
        one_hot=True,
        random_flip=False,
        random_shift=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)

    # test_dataset = dataset_def(
    #     cfg.DATA.NUM_SAMPLES,
    #     split=cfg.TEST.DATASET,
    #     random_flip=False,
    #     random_shift=False,
    #     one_hot=True,
    #     from_rgb_detection=cfg.FROM_RGB_DET,
    #     overwritten_data_path=cfg.OVER_WRITE_TEST_FILE)

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=cfg.NUM_WORKERS,
    #     pin_memory=True,
    #     drop_last=False,
    #     collate_fn=collate_fn)

    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else 4
    NUM_VEC = 0 if cfg.DATA.CAR_ONLY else 3
    NUM_CLASSES = len(cfg.MODEL.CLASSES)

    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)

    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    if os.path.isfile(cfg.TEST.WEIGHTS):
        checkpoint = torch.load(cfg.TEST.WEIGHTS)
        # start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        # best_epoch = checkpoint['best_epoch']
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TEST.WEIGHTS, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("=> loaded checkpoint '{}')".format(cfg.TEST.WEIGHTS))
    else:
        logging.error("=> no checkpoint found at '{}'".format(cfg.TEST.WEIGHTS))
        assert False

    save_file_name = os.path.join(SAVE_DIR, 'detection.pkl')
    result_folder = os.path.join(SAVE_DIR, 'result')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    det_results = test(model, test_dataset, test_loader, save_file_name, classes_mapper)

    # Write detection results for KITTI evaluation
    print(len(det_results))
    if cfg.TEST.METHOD == 'nms':
        write_detection_results_nms(result_folder, cfg.DATA.TEST_DATASET_PATHS, det_results)
    else:
        write_detection_results(result_folder, cfg.DATA.TEST_DATASET_PATHS, det_results)

    output_dir = os.path.join(result_folder, 'data')

    if 'test' not in cfg.TEST.DATASET:
        # evaluate_py_wrapper(result_folder, cfg.DATA.TEST_DATASET_PATHS)
        evaluate_cuda_wrapper(output_dir,cfg.DATA.TEST_DATASET_PATHS, classes_mapper)
    else:
        logger.info('results file save in  {}'.format(result_folder))
        os.system('cd %s && zip -q -r ../results.zip *' % (result_folder))


# car AP(Average Precision)@0.70, 0.70, 0.70:
# bbox AP:11.46, 11.47, 11.47
# bev  AP:0.10, 0.10, 0.10
# 3d   AP:0.01, 0.01, 0.01
# car AP(Average Precision)@0.70, 0.50, 0.50:
# bbox AP:11.46, 11.47, 11.47
# bev  AP:0.19, 0.19, 0.19
# 3d   AP:0.05, 0.05, 0.05
# pedestrian AP(Average Precision)@0.50, 0.50, 0.50:
# bbox AP:4.55, 4.55, 4.55
# bev  AP:0.03, 0.03, 0.03
# 3d   AP:0.02, 0.02, 0.02
# pedestrian AP(Average Precision)@0.50, 0.25, 0.25:
# bbox AP:4.55, 4.55, 4.55
# bev  AP:0.61, 0.61, 0.61
# 3d   AP:0.08, 0.08, 0.08