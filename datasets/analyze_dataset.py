import sys
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from  kitti import kitti_util as utils


DATASET_PATH = '/dataset/kitti_format/lyft/train/'

label_dir = os.path.join(DATASET_PATH, 'label_2')
file_paths = [f.path for f in os.scandir(label_dir) if f.is_file() and f.name.endswith('.txt')]

objects_data = {}

for fp in file_paths:
    objects = utils.read_label(fp)
    for obj in objects:
        if obj.type in objects_data:
            objects_data[obj.type]['n']+=1
            objects_data[obj.type]['l']+=obj.l
            objects_data[obj.type]['w']+=obj.w
            objects_data[obj.type]['h']+=obj.h
        else:
            objects_data[obj.type] = {'n':1,
                                      'l':obj.l,
                                      'w':obj.w,
                                      'h':obj.h}
for class_n, info in objects_data.items():
    objects_data[class_n]['l']/=objects_data[class_n]['n']
    objects_data[class_n]['w']/=objects_data[class_n]['n']
    objects_data[class_n]['h']/=objects_data[class_n]['n']

    print('{} (n_objects: {})'.format(class_n, objects_data[class_n]['n']))
    print('l: {} w: {} h: {}'.format(objects_data[class_n]['l'],
                                     objects_data[class_n]['w'],
                                     objects_data[class_n]['h']))
            
