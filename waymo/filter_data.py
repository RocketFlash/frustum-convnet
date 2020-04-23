import numpy as np
import math
import sys
import os
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import kitti.kitti_util as utils
import waymo.waymo_utils as wutils

def bounding_box_filter(points, x_range, y_range, z_range):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return points[bb_filter]


def filter_boxes_kitti(objects, lidar_scan, calib, thresh=5):
    lidar_scan_new = np.copy(lidar_scan)
    lidar_scan_new[:,3] = 1
    
    Tr_V2C = np.eye(4)
    Tr_V2C[:3,:] = calib.V2C
    
    lidar_scan_new = np.matmul(Tr_V2C, lidar_scan_new.T)
    lidar_scan_new=lidar_scan_new.T
    
    # print('Num objects before filtering {}'.format(len(objects)))
    filtered_objects = []
    for obj in objects:
        Tr_2box = wutils.get_box_transformation_matrix_kitti(obj.t, (obj.l, obj.h, obj.w), obj.ry)
        Tr_2box_inv = np.linalg.inv(Tr_2box)
        lidar_scan_box = np.matmul(Tr_2box_inv, lidar_scan_new.T)
        lidar_scan_box = lidar_scan_box.T

        x_range = [-1/2, 1/2]
        y_range = [0, 1]
        z_range = [-1/2, 1/2]
        filtered_points = bounding_box_filter(lidar_scan_box[:,:3], x_range, y_range, z_range)
        
        if len(filtered_points)>=thresh:
            filtered_objects.append(obj)
    # print('Num objects after filtering {}'.format(len(filtered_objects)))       
    return filtered_objects


def filter_boxes(objects, lidar_scan, thresh=5):
    lidar_scan_new = np.copy(lidar_scan)
    lidar_scan_new[:,3] = 1
    
    # print('Num objects before filtering {}'.format(len(objects)))
    filtered_objects = []
    for obj in objects:
        Tr_2box = wutils.get_box_transformation_matrix(obj.t, (obj.l, obj.h, obj.w), obj.ry)
        Tr_2box_inv = np.linalg.inv(Tr_2box)
        lidar_scan_box = np.matmul(Tr_2box_inv, lidar_scan_new.T)
        lidar_scan_box = lidar_scan_box.T

        x_range = [-1/2, 1/2]
        y_range = [-1/2, 1/2]
        z_range = [0, 1]
        filtered_points = bounding_box_filter(lidar_scan_box[:,:3], x_range, y_range, z_range)
        
        if len(filtered_points)>=thresh:
            filtered_objects.append(obj)
    # print('Num objects after filtering {}'.format(len(filtered_objects)))       
    return filtered_objects


def main():
    gt_root_folder = '/dataset/kitti_format/waymo/training/'
    cam_idx = 0
    thresh = 5
    lidar_dtype=float

    gt_label_folder = os.path.join(gt_root_folder,'label_{}/'.format(cam_idx))
    gt_lidar_folder = os.path.join(gt_root_folder,'velodyne/')
    gt_calib_folder = os.path.join(gt_root_folder,'calib/')
    new_label_folder = os.path.join(gt_root_folder,'label_filtered_{}/'.format(cam_idx))
    os.makedirs(new_label_folder, exist_ok=True) 

    label_names = [f.name.split('.')[0] for f in os.scandir(gt_label_folder ) if f.is_file() and f.name.endswith('.txt')]
    for sample_name in tqdm.tqdm(label_names):
        calib_filename = os.path.join(gt_calib_folder, sample_name+'.txt')
        gt_filename = os.path.join(gt_label_folder, sample_name+'.txt')
        lidar_filename = os.path.join(gt_lidar_folder, sample_name+'.bin')
        
        lidar_scan = utils.load_velo_scan(lidar_filename, dtype=lidar_dtype)
        
        calib = utils.Calibration(calib_filename, cam_idx=cam_idx)
        objects_gt = utils.read_label(gt_filename)
        filtered_objects = filter_boxes_kitti(objects_gt, lidar_scan, calib, thresh=thresh)
        writepath = os.path.join(new_label_folder, sample_name+'.txt')

        # print(f"Before filtering: {len(objects_gt)}, after filtering {len(filtered_objects)}")
        with open(writepath, 'w') as f:
            for obj in filtered_objects:
                f.write(obj.get_string_ann()+'\n')

if __name__ == '__main__':
    main()

    