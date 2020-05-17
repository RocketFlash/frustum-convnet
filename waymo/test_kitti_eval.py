import sys
import numpy as np
KITTI_EVAL_PATH = '../../kitti-object-eval-python/'
sys.path.append(KITTI_EVAL_PATH)

import kitti_common as kitti
from eval import get_coco_eval_result, get_official_eval_result

class_name = 'VEHICLE'
x1, y1, x2, y2 = 100, 100, 400, 400
tx, ty, tz = 1, 1, 1
h, w, l = 2, 2, 2
ry = 1
score = 1
content = [[class_name, x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry, score]]
annotations = {}
annotations.update({
    'name': [],
    'truncated': [],
    'occluded': [],
    'alpha': [],
    'bbox': [],
    'dimensions': [],
    'location': [],
    'rotation_y': []
})

annotations['name'] = np.array([x[0] for x in content])
annotations['truncated'] = np.array([float(0) for x in content])
annotations['occluded'] = np.array([int(0) for x in content])
annotations['alpha'] = np.array([float(1) for x in content])
annotations['bbox'] = np.array([[float(info) for info in x[1:5]] for x in content]).reshape(-1, 4)
# dimensions will convert hwl format to standard lhw(camera) format.
annotations['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
# annotations['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)
annotations['location'] = np.array([[float(info) for info in x[5:8]] for x in content]).reshape(-1, 3)
annotations['rotation_y'] = np.array([float(x[11]) for x in content]).reshape(-1)
if len(content) != 0 and len(content[0]) == 13:  # have score
    annotations['score'] = np.array([float(x[12]) for x in content])
else:
    annotations['score'] = np.zeros([len(annotations['bbox'])])

gt_annos = [annotations]
det_annos = [annotations]

print(get_official_eval_result(gt_annos, det_annos, ['VEHICLE','PEDESTRIAN']))