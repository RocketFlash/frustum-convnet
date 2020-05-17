import numpy as np

class KITTICategory(object):
    
    # Class: car (n_objects: 461510)
    # l: 4.561454364996091 w: 1.8644759318733815 h: 1.7098814750154965
    # Class: pedestrian (n_objects: 104329)
    # l: 0.7883843290673025 w: 0.7125122851218032 h: 1.7814643944351918
    # Class: twowheels (n_objects: 22488)
    # l: 1.7687252728482616 w: 0.6156853149645731 h: 1.4566547263222673
    CLASSES = ['car', 'pedestrian', 'twowheels']
    CLASS_MEAN_SIZE = {
        'car': np.array([4.561454364996091, 1.8644759318733815, 1.7098814750154965]),
        'pedestrian': np.array([0.7883843290673025, 0.7125122851218032, 1.7814643944351918]),
        'twowheels': np.array([1.7687252728482616, 0.6156853149645731, 1.4566547263222673]),
    }
 
    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]


class WaymoCategory(object):

    CLASSES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    
    CLASS_MEAN_SIZE = {
        'VEHICLE': np.array([4.561454364996091, 1.8644759318733815, 1.7098814750154965]),
        'PEDESTRIAN': np.array([0.7883843290673025, 0.7125122851218032, 1.7814643944351918]),
        'CYCLIST': np.array([1.7687252728482616, 0.6156853149645731, 1.4566547263222673]),
    }
 
    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]


DATASET_INFO = {
    "KITTI": KITTICategory,
    "Waymo": WaymoCategory
}