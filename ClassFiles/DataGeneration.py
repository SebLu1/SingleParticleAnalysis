import mrcfile
import numpy as np
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
from ClassFiles.ut import getRecos, locate_gt
import random

DEFAULT_BATCH_SIZE = 1


def get_dict(noise_levels, methodes, eval_data, data_path=None):
    train_dict = {}
    for nl in noise_levels:
        train_dict[nl] = {}
        for met in methodes:
            if met == 'EM':
                train_dict[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='All', data_path=data_path)
            elif met == 'SGD':
                train_dict[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='Final', data_path=data_path)
            else:
                raise ValueError('Enter valid noise level')
    return train_dict


def get_image(noise_level, method, data_dict):
    data_list = data_dict[noise_level][method]
    adv_path = random.choice(data_list)

    with mrcfile.open(adv_path) as mrc:
        adv = mrc.data
    with mrcfile.open(locate_gt(adv_path)) as mrc:
        gt = mrc.data
    return gt, adv


def get_batch(noise_levels, methods, batch_size=DEFAULT_BATCH_SIZE,
              data_dict=None):
    true = np.zeros(shape=(batch_size, 96, 96, 96))
    adv = np.zeros(shape=(batch_size, 96, 96, 96))
    for k in range(batch_size):
        nl = random.choice(noise_levels)
        method = random.choice(methods)
        gt, adver = get_image(nl, method, data_dict)
        true[k, ...] = gt
        adv[k, ...] = adver
    return true, adv
