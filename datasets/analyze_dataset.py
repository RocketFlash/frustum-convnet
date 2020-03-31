import sys
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from  kitti import kitti_util as utils


# DATASET_PATH = '/dataset/kitti_format/lyft/train/'
# DATASET_PATH = '/dataset/kitti_format/argo/train/'
DATASET_PATHS = ['/dataset/kitti_format/lyft/train/',
                 '/dataset/kitti_format/kitti/training/',
                 '/dataset/kitti_format/argo/train/']

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

objects_all = {'lyft':{},
               'kitti':{},
               'argo':{}}
for DATASET_PATH in DATASET_PATHS:
    dataset_name = DATASET_PATH.split('/')[-3]
    label_dir = os.path.join(DATASET_PATH, 'label_2')
    file_paths = [f.path for f in os.scandir(label_dir) if f.is_file() and f.name.endswith('.txt')]

    objects_data = {}

    for fp in file_paths:
        objects = utils.read_label(fp)
        for obj in objects:
            if obj.type not in classes_mapper:
                continue
            if classes_mapper[obj.type] in objects_data:
                objects_data[classes_mapper[obj.type]]['n']+=1
                objects_data[classes_mapper[obj.type]]['l']+=obj.l
                objects_data[classes_mapper[obj.type]]['w']+=obj.w
                objects_data[classes_mapper[obj.type]]['h']+=obj.h
            else:
                objects_data[classes_mapper[obj.type]] = {'n':1,
                                        'l':obj.l,
                                        'w':obj.w,
                                        'h':obj.h}

    print('############################## {} ###############################'.format(dataset_name))
    for class_n, info in objects_data.items():
        objects_data[class_n]['l']/=objects_data[class_n]['n']
        objects_data[class_n]['w']/=objects_data[class_n]['n']
        objects_data[class_n]['h']/=objects_data[class_n]['n']
        
        print('{} (n_objects: {})'.format(class_n, objects_data[class_n]['n']))
        print('l: {} w: {} h: {}'.format(objects_data[class_n]['l'],
                                        objects_data[class_n]['w'],
                                        objects_data[class_n]['h']))
    objects_all[dataset_name] = objects_data


mean_sizes = {}
for dataset_name, data in objects_all.items():
    for class_n, info in data.items():
        if class_n in mean_sizes:
            mean_sizes[class_n]['n']+=info['n']
            mean_sizes[class_n]['l']+=info['l']/len(objects_all)
            mean_sizes[class_n]['w']+=info['w']/len(objects_all)
            mean_sizes[class_n]['h']+=info['h']/len(objects_all)
        else:
            mean_sizes[class_n] = info
            mean_sizes[class_n]['l']=info['l']/len(objects_all)
            mean_sizes[class_n]['w']=info['w']/len(objects_all)
            mean_sizes[class_n]['h']=info['h']/len(objects_all)

print('############################## MERGED DATASET ###############################')
for class_n, info in mean_sizes.items():
    print('Class: {} (n_objects: {})'.format(class_n, info['n']))
    print('l: {} w: {} h: {}'.format(info['l'], info['w'], info['h']))
# ############################## lyft ###############################
# car (n_objects: 166096)
# l: 4.908415916096603 w: 1.9485082121183266 h: 1.763952172237821
# pedestrian (n_objects: 6999)
# l: 0.8095799399914204 w: 0.771495927989713 h: 1.7755679382768552
# twowheels (n_objects: 6266)
# l: 1.7920922438557352 w: 0.6472741781040598 h: 1.4554691988509494
# ############################## kitti ###############################
# car (n_objects: 32750)
# l: 4.198177404580118 w: 1.6848757251907918 h: 1.6442769465649043
# pedestrian (n_objects: 4487)
# l: 0.8422843770893668 w: 0.6601894361488833 h: 1.7607064854022891
# twowheels (n_objects: 1627)
# l: 1.76354640442532 w: 0.5967732022126637 h: 1.7372034419176359
# ############################## argo ###############################
# car (n_objects: 262664)
# l: 4.577769774311551 w: 1.9600438583110262 h: 1.7214153062437636
# twowheels (n_objects: 14595)
# l: 1.7505371702637293 w: 0.6030085645769958 h: 1.177291538198217
# pedestrian (n_objects: 92843)
# l: 0.7132886701211203 w: 0.7058514912268132 h: 1.8081187596264312


# LYFT
# car (n_objects: 161818)
# l: 4.764630078235898 w: 1.9247089940550761 h: 1.719876651546986
# other_vehicle (n_objects: 10695)
# l: 8.219887798036513 w: 2.7861692379616674 h: 3.2447498831229615
# bus (n_objects: 2874)
# l: 12.513347251217802 w: 2.961562282533075 h: 3.438239387613066
# truck (n_objects: 4278)
# l: 10.347204301075266 w: 2.8487283777466104 h: 3.4311360448808053
# pedestrian (n_objects: 6999)
# l: 0.8095799399914204 w: 0.771495927989713 h: 1.7755679382768552
# bicycle (n_objects: 6010)
# l: 1.767893510815312 w: 0.6343427620632334 h: 1.4498286189683964
# motorcycle (n_objects: 256)
# l: 2.360195312500003 w: 0.9508593750000002 h: 1.5878906249999987
# animal (n_objects: 51)
# l: 0.7237254901960788 w: 0.37627450980392174 h: 0.5256862745098038
# emergency_vehicle (n_objects: 39)
# l: 7.158717948717953 w: 2.5417948717948726 h: 2.435897435897437

#ARGO
# VEHICLE (n_objects: 253845)
# l: 4.496331737869433 w: 1.9345668813623207 h: 1.6795860466002868
# LARGE_VEHICLE (n_objects: 8819)
# l: 6.921872094342087 w: 2.693369996598254 h: 2.9254235174055525
# ON_ROAD_OBSTACLE (n_objects: 31603)
# l: 0.5208752333639142 w: 0.665371958358376 h: 1.082935480808811
# BICYCLE (n_objects: 11914)
# l: 1.7191002182305852 w: 0.5706899446030569 h: 1.155207319120501
# BICYCLIST (n_objects: 7819)
# l: 1.549999999999851 w: 0.8919209617598293 h: 1.8438086711855939
# MOTORCYCLIST (n_objects: 1651)
# l: 1.1910054512416564 w: 0.7858086008479802 h: 1.642901271956403
# MOTORCYCLE (n_objects: 1422)
# l: 2.191476793248952 w: 0.8254922644163123 h: 1.342215189873425
# PEDESTRIAN (n_objects: 92843)
# l: 0.7132886701211203 w: 0.7058514912268132 h: 1.8081187596264312
# BUS (n_objects: 1332)
# l: 11.6261036036035 w: 2.8639714714714466 h: 3.2536186186186224
# ANIMAL (n_objects: 137)
# l: 1.1777372262773742 w: 1.0 h: 1.0
# OTHER_MOVER (n_objects: 2361)
# l: 1.3015036001694327 w: 0.9355061414654692 h: 1.256480304955524
# MOPED (n_objects: 1259)
# l: 1.5499999999999685 w: 0.6575536139793494 h: 1.2000000000000273
# STROLLER (n_objects: 520)
# l: 0.9384807692307627 w: 0.62130769230769 h: 1.3320576923076963
# TRAILER (n_objects: 1785)
# l: 14.34568627450977 w: 3.0381400560223972 h: 3.4465098039215634
# EMERGENCY_VEHICLE (n_objects: 506)
# l: 4.970454545454536 w: 1.8309090909090846 h: 1.6060474308300352

#KITTI
# Car (n_objects: 28742)
# l: 3.883954491684635 w: 1.6285898684851166 h: 1.5260834319115126
# Van (n_objects: 2914)
# l: 5.078366506520258 w: 1.9020796156485942 h: 2.20659231297187
# DontCare (n_objects: 11295)
# l: -1.0 w: -1.0 h: -1.0
# Tram (n_objects: 511)
# l: 16.09426614481405 w: 2.5437377690802285 h: 3.528923679060668
# Truck (n_objects: 1094)
# l: 10.109076782449753 w: 2.5850914076782585 h: 3.2517093235831673
# Pedestrian (n_objects: 4487)
# l: 0.8422843770893668 w: 0.6601894361488833 h: 1.7607064854022891
# Cyclist (n_objects: 1627)
# l: 1.76354640442532 w: 0.5967732022126637 h: 1.7372034419176359
# Person_sitting (n_objects: 222)
# l: 0.8020270270270264 w: 0.5949099099099106 h: 1.274954954954955
# Misc (n_objects: 973)
# l: 3.5755909558067804 w: 1.5138335046248708 h: 1.9071325796505694