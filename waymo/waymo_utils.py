import numpy as np
import cv2
import math
import os
import tqdm
from waymo_open_dataset.utils import frame_utils
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def get_sample_names(dataset_path):
    dataset_idxs = []
    tfrecord_filenames = [f.path for f in os.scandir(dataset_path) if f.is_file() and f.name.endswith('.tfrecord')]
    datasets = []
    for dataset_idx, file_name in enumerate(tqdm.tqdm(tfrecord_filenames)):
        dataset = tf.data.TFRecordDataset(file_name, compression_type='')
        datasets.append(dataset)
        for idx, _ in enumerate(dataset):
            dataset_idxs.append((dataset_idx, idx))
    return datasets, dataset_idxs


def get_box_transformation_matrix(obj_loc, obj_size, obj_ry):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = obj_loc
    c = math.cos(obj_ry)
    s = math.sin(obj_ry)

    sl, sh, sw = obj_size

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])


def get_box_transformation_matrix_kitti(obj_loc, obj_size, ry):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = obj_loc
    c = math.cos(ry)
    s = math.sin(ry)

    sl, sh, sw = obj_size

    return np.array([
        [       -sl*c,    0,  sw*s,    tx],
        [          0,  -sh,    0,      ty],
        [    -sl*(-s),    0,  sw*c,    tz],
        [          0,    0,    0,     1]])


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_images(frame):
    """ parse and save the images in png format
            :param frame: open dataset frame proto
    """
    images = [None] * 5
    for img in frame.images:
        cam_idx = img.name - 1
        img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images[cam_idx] = rgb_img
    return images


def get_calibs(frame):
    """ parse and save the calibration data
            :param frame: open dataset frame proto
            :param frame_num: the current frame number
            :return:
    """
    waymo_cam_RT=np.array([[0,-1 ,0 ,0],
                           [0 ,0 ,-1,0],
                           [1 ,0 ,0 ,0],
                           [0 ,0 ,0 ,1]])
    camera_calib = [None] * 5
    Tr_velo_to_cam = [None] * 5
    R0_rect =  np.eye(3)
    calibs = []

    for camera in frame.context.camera_calibrations:
        tmp=np.array(camera.extrinsic.transform).reshape(4,4)
        tmp=np.linalg.inv(tmp)
        tmp = np.matmul(waymo_cam_RT, tmp)
        Tr_velo_to_cam[camera.name-1] = tmp

    for cam in frame.context.camera_calibrations:
        tmp=np.zeros((3,4))
        tmp[0,0]=cam.intrinsic[0]
        tmp[1,1]=cam.intrinsic[1]
        tmp[0,2]=cam.intrinsic[2]
        tmp[1,2]=cam.intrinsic[3]
        tmp[2,2]=1
        camera_calib[cam.name-1] = tmp
    
    for Tr, P in zip(Tr_velo_to_cam, camera_calib):
        calibs.append(Calibration(Tr, P, R0_rect))
    return calibs


def get_lidar(frame):
    """ parse and save the lidar data in psd format
            :param frame: open dataset frame proto
            """
    range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)

    points_all = np.concatenate(points, axis=0)
    intensity_all = np.ones((points_all.shape[0],1))
    point_cloud = np.column_stack((points_all, intensity_all))

    return point_cloud


def get_label(frame, calibs):
    """ parse and save the label data in .txt format
            :param frame: open dataset frame proto
            """
    # preprocess bounding box data
    type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
    lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']

    id_to_bbox = dict()
    id_to_name = dict()
    for labels in frame.projected_lidar_labels:
        name = labels.name
        print(f'camera name {name}')
        for label in labels.labels:
            bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
            id_to_bbox[label.id] = bbox
            id_to_name[label.id] = name - 1
            print(f'label_id {label.id}')

    Tr_velo_to_cam = [calib.V2C for calib in calibs]
    objects = {}

    for obj in frame.laser_labels:
        # caculate bounding box
        bounding_box = None
        name = None
        obj_id = obj.id
        print(f'obj_id {obj_id}')
        for lidar in lidar_list:
            print(f'obj_id + lidar {obj_id + lidar}')
            if obj_id + lidar in id_to_bbox:
                bounding_box = id_to_bbox.get(obj_id + lidar)
                name = str(id_to_name.get(obj_id + lidar))

                if int(name) not in objects:
                    objects[int(name)] = []
                my_type = type_list[obj.type]
                truncated = 0
                occluded = 0
                height = obj.box.height
                width = obj.box.width
                length = obj.box.length
                x = obj.box.center_x
                y = obj.box.center_y
                z = obj.box.center_z - height/2
                rotation_y = obj.box.heading
                
                transform_box_to_cam = Tr_velo_to_cam[int(name)] @ get_box_transformation_matrix((x, y, z),(length,height,width), rotation_y)
                pt1 = np.array([-0.5, 0.5, 0 , 1.])
                pt2 = np.array([0.5, 0.5, 0 , 1.])
                pt1 = np.matmul(transform_box_to_cam, pt1)
                pt2 = np.matmul(transform_box_to_cam, pt2)
                new_ry = math.atan2(pt2[2]-pt1[2],pt2[0]-pt1[0])
                rotation_y = -new_ry

                new_loc = np.matmul(Tr_velo_to_cam[int(name)], np.array([x,y,z,1]).T)
                x, y, z = new_loc[:3]

                beta = math.atan2(x, z)
                alpha = (rotation_y + beta - math.pi / 2) % (2 * math.pi)

                obj_params = {'type' : my_type,
                            'truncation' : round(truncated, 2),
                            'occlusion' : int(occluded),
                            'alpha' : round(alpha, 2),
                            'xmin' : round(bounding_box[0], 2),
                            'ymin' : round(bounding_box[1], 2),
                            'xmax' : round(bounding_box[2], 2),
                            'ymax' : round(bounding_box[3], 2),
                            'h' : round(height, 2),
                            'w' : round(width, 2),
                            'l' : round(length, 2),
                            't' : (round(x, 2), round(y, 2), round(z, 2)),
                            'ry' : round(rotation_y, 2),
                            'score' : 1
                            }
                objects[int(name)].append(Object3d(obj_params))

    return objects


def compute_2d_bounding_box(img_or_shape,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape,tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)


def get_labels_in_cam(frame, calibs, image_shapes):
    """ parse and save the label data in .txt format
            :param frame: open dataset frame proto
            """
    # preprocess bounding box data
    type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
    lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
    offset = 100

    Tr_velo_to_cam = [calib.V2C for calib in calibs]
    vehicle_to_images = []
    for calib in calibs:
        P_mat = np.eye(4)
        P_mat[:3,:] = calib.P

        Tr_V2C = np.eye(4)
        Tr_V2C[:3,:] = calib.V2C
        vehicle_to_images.append(np.matmul(P_mat, Tr_V2C))

    objects = {}

    for obj in frame.laser_labels:
        # caculate bounding box
        box = obj.box

        for name in range(5):
            box_to_vehicle = get_box_transformation_matrix((box.center_x,box.center_y,box.center_z), (box.length, box.height, box.width), box.heading)

            box_to_image = np.matmul(vehicle_to_images[name] , box_to_vehicle)
            # Loop through the 8 corners constituting the 3D box
            # and project them onto the image

            vertices = np.empty([2,2,2,2])
            skip_box = False
            for k in [0, 1]:
                if skip_box:
                    break
                for l in [0, 1]:
                    if skip_box:
                        break
                    for m in [0, 1]:
                        # 3D point in the box space
                        v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                        # Project the point onto the image
                        v = np.matmul(box_to_image, v)

                        # If any of the corner is behind the camera, ignore this object.
                        if v[2] < 0:
                            skip_box = True
                            break

                        vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]
            if skip_box:
                continue

            vertices = vertices.astype(np.int32)
            x1,y1,x2,y2 = compute_2d_bounding_box(image_shapes[name], vertices)

            if (x1 > image_shapes[name][1] - 1 - offset) or (x2 < offset) or (y1 > image_shapes[name][0] - 1 - offset) or (y2 < offset):
               continue
            bounding_box = [x1, y1, x2, y2]

            if int(name) not in objects:
                objects[int(name)] = []
            my_type = type_list[obj.type]
            truncated = 0
            occluded = 0
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height/2
            rotation_y = obj.box.heading
            
            transform_box_to_cam = Tr_velo_to_cam[int(name)] @ get_box_transformation_matrix((x, y, z),(length,height,width), rotation_y)
            pt1 = np.array([-0.5, 0.5, 0 , 1.])
            pt2 = np.array([0.5, 0.5, 0 , 1.])
            pt1 = np.matmul(transform_box_to_cam, pt1)
            pt2 = np.matmul(transform_box_to_cam, pt2)
            new_ry = math.atan2(pt2[2]-pt1[2],pt2[0]-pt1[0])
            rotation_y = -new_ry

            new_loc = np.matmul(Tr_velo_to_cam[int(name)], np.array([x,y,z,1]).T)
            x, y, z = new_loc[:3]

            beta = math.atan2(x, z)
            alpha = (rotation_y + beta - math.pi / 2) % (2 * math.pi)

            obj_params = {'type' : my_type,
                        'truncation' : round(truncated, 2),
                        'occlusion' : int(occluded),
                        'alpha' : round(alpha, 2),
                        'xmin' : round(bounding_box[0], 2),
                        'ymin' : round(bounding_box[1], 2),
                        'xmax' : round(bounding_box[2], 2),
                        'ymax' : round(bounding_box[3], 2),
                        'h' : round(height, 2),
                        'w' : round(width, 2),
                        'l' : round(length, 2),
                        't' : (round(x, 2), round(y, 2), round(z, 2)),
                        'ry' : round(rotation_y, 2),
                        'score' : 1
                        }
            objects[int(name)].append(Object3d(obj_params))
    return objects
      

class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, V2C, P, R0):
        # Rigid transform from Velodyne coord to reference camera coord
        self.P = P
        self.V2C = V2C
        self.V2C = self.V2C[:3,:]

        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = R0

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)
        self.b_y = self.P[1, 3] / (-self.f_v) 

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        """
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        # print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        # print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            depth_box_fov_inds = (
                (depth_pc_velo[:, 0] < cbox[0][1])
                & (depth_pc_velo[:, 0] >= cbox[0][0])
                & (depth_pc_velo[:, 1] < cbox[1][1])
                & (depth_pc_velo[:, 1] >= cbox[1][0])
                & (depth_pc_velo[:, 2] < cbox[2][1])
                & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, obj_params):
        
        # extract label, truncation, occlusion
        self.type = obj_params['type']  # 'Car', 'Pedestrian', ...
        self.truncation = obj_params['truncation']  # truncated pixel ratio [0..1]
        self.occlusion = obj_params['occlusion']  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = obj_params['alpha']  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = obj_params['xmin']  # left
        self.ymin = obj_params['ymin']  # top
        self.xmax = obj_params['xmax']  # right
        self.ymax = obj_params['ymax']  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = obj_params['h']  # box height
        self.w = obj_params['w']  # box width
        self.l = obj_params['l']  # box length (in meters)
        self.t = obj_params['t']  # location (x,y,z) in camera coord.
        self.ry = obj_params['ry']  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.score = obj_params['score']

    def print_object(self):
        print('Type: %s, truncation: %d, occlusion: %d, alpha: %f' %
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' %
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' %
              (self.t[0], self.t[1], self.t[2], self.ry))

    def get_string_ann(self):
        output_str = self.type + " %d %d %.6f " % (self.truncation, self.occlusion, self.alpha)
        output_str += "%.6f %.6f %.6f %.6f " % (self.xmin, self.ymin, self.xmax, self.ymax)
        output_str += "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (self.h, self.w, self.l,
                                                                   self.t[0], self.t[1], self.t[2], self.ry, self.score)

        return output_str
