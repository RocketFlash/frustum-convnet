import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append('/root/workdir/waymo_challenge/frustum-convnet')

import pandas as pd
import csv
import numpy as np
import tqdm
import cv2
from predict_on_frame import predict_frame
from train.inference import load_model
import tqdm 



DATASET_PATH = '/dataset/kitti_format/waymo/waymo_challenge_3/validation/'
PRED_PATH = DATASET_PATH + 'preds_3/'
PRED_FILE = PRED_PATH + 'pred_all.csv'
INFOS_PATH = DATASET_PATH + 'infos/'
INFOS_FILE = INFOS_PATH + 'frames_info_all.csv'
CFG_FILE_PATH = 'cfgs/det_sample_waymo.yaml'
CONFIDENCE_THRESH = 0.3
SAVE_EACH = 10
NUM_GPUS = 1
TFRECORDS_PATH = '/dataset/waymo/validation/'
SAVE_PRED_PATH = '/root/workdir/waymo_challenge/predictions/'

predictions_df = pd.read_csv(PRED_FILE)  
total_num_frames = predictions_df.shape[0]
infos_df = pd.read_csv(INFOS_FILE)

class_to_idx = {'VEHICLE': 0,
                'PEDESTRIAN': 1,
                'CYCLIST': 2}

model = load_model(CFG_FILE_PATH, testing =True)

if NUM_GPUS > 1:
    model = torch.nn.DataParallel(model, device_ids = [0, 1])

predictions_df['tfrec_frame_num'] = predictions_df.apply(lambda row: '_'.join(row['image_name'].split('_')[:2]), axis=1)
predictions_df['tfrecord_name'] = infos_df['tfrecord_name']

predictions_grouped = predictions_df.groupby(predictions_df.tfrec_frame_num)
pred_groups = predictions_grouped.groups.keys()

preds_to_csv = []
preds_to_csv_all = []
chunk_idx = 0


for index, group in enumerate(tqdm.tqdm(pred_groups)):
    frame_data = predictions_grouped.get_group(group)
    tfrec_idx = int(group.split('_')[0])
    frame_idx = int(group.split('_')[-1])
    to_frustum_bboxes = [[]] * 5
    meta_infos = []
    for index_i, row in frame_data.iterrows():
        bboxes_i = []
        image_name = row['image_name']
        context_name = row['context']
        timestamp = row['timestamp']

        if isinstance(row['bboxes'], str):
            bboxes = row['bboxes'].split(',')
            if len(bboxes)==1 and bboxes[0] == '':
                continue
            bboxes_float = [[float(bbox_el) for bbox_el in bbox[1:-1].split(' ')] for bbox in bboxes]
            
            image_name_without_ext = image_name.split('.')[0]
            camera_name = image_name_without_ext.split('_')[-1]
            tfrecord_name = row['tfrecord_name']
            camera_idx = int(camera_name)

            for i, bbox in enumerate(bboxes_float):
                center_x, center_y, w, h, pred_class, score = bbox
                if score>CONFIDENCE_THRESH:
                    x1, y1, x2, y2 = int(center_x - w/2), int(center_y - h/2), int(center_x + w/2), int(center_y + h/2)
                    if pred_class == 1:
                        class_name = 'VEHICLE'
                    elif pred_class == 2:
                        class_name = 'PEDESTRIAN'
                    elif pred_class == 4:
                        class_name = 'CYCLIST'
                    else:
                        continue
                    bboxes_i.append([x1, y1, x2, y2, class_name, score])


        to_frustum_bboxes[camera_idx-1] = bboxes_i


    predictions_all, datas_all = predict_frame(model, TFRECORDS_PATH + tfrecord_name, frame_idx, to_frustum_bboxes)
    
    preds_str = ''
    for cam_idx in range(len(predictions_all)):
        if predictions_all[cam_idx] is not None:
            # print(f'Cam idx: {cam_idx} , num predictions: {len(predictions_all[cam_idx])}')
            for data, pred in zip(datas_all[cam_idx], predictions_all[cam_idx]):
                if data is None:
                    continue

                class_type = data['type_i']
                box2d = data['box2d_i']
                x1, y1, x2, y2 = box2d

                for pred_n in pred:  
                    h, w, l, tx, ty, tz, ry, score = pred_n
                    preds_str += f'[{cam_idx+1} {class_to_idx[class_type]} {x1} {y1} {x2} {y2} {tx} {ty} {tz} {h} {w} {l} {ry} {score}],'
        
    preds_str = preds_str[:-1]
    preds_to_csv.append([tfrecord_name, tfrec_idx, frame_idx, preds_str])
    preds_to_csv_all.append([tfrecord_name, tfrec_idx, frame_idx, preds_str])

    if index%SAVE_EACH==0:
        df = pd.DataFrame(preds_to_csv, columns =['tfrecord_name', 'tfrec_idx', 'frame_idx', 'predictions'])
        df.to_csv(f'{SAVE_PRED_PATH}pred_{chunk_idx}.csv',  index=False)
        preds_to_csv = []
        chunk_idx+=1

        
df = pd.DataFrame(preds_to_csv_all, columns =['tfrecord_name', 'tfrec_idx', 'frame_idx', 'predictions'])
df.to_csv(f'{SAVE_PRED_PATH}pred_all.csv',  index=False)
        
        

