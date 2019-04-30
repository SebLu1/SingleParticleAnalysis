import mrcfile
import numpy as np
from ClassFiles.ut import getRecos, locate_gt
import random

# The path to find corresponding ground truth images
GT_PATH = '/local/scratch/public/sl767/MRC_Data/org/'

NOISE_LEVELS = ['01', '016', '02']
METHODS = ['EM', 'SGD']
BATCH_SIZE = 1


def get_dic(noise_levels, methodes, eval_data):
    train_dic = {}
    for nl in noise_levels:
        train_dic[nl] = {}
        for met in methodes:
            if met == 'EM':
                train_dic[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='All')
            elif met == 'SGD':
                train_dic[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='Final')
            else:
                raise ValueError('Enter valid noise level')
    return train_dic


train_dic = get_dic(NOISE_LEVELS, METHODS, eval_data=False)
eval_dic = get_dic(NOISE_LEVELS, METHODS, eval_data=True)


def get_image(noise_level, methode, eval_data):
    if eval_data:
        d = eval_dic
    else:
        d = train_dic
    l = d[noise_level][methode]
    adv_path = random.choice(l)

    with mrcfile.open(adv_path) as mrc:
        adv = mrc.data
    with mrcfile.open(locate_gt(adv_path)) as mrc:
        gt = mrc.data
    return gt, adv


def get_batch(batch_size=BATCH_SIZE, noise_levels=NOISE_LEVELS, methods=METHODS, eval_data=False):
    true = np.zeros(shape=(batch_size, 96,96,96))
    adv = np.zeros(shape=(batch_size, 96,96,96))
    for k in range(BATCH_SIZE):
        nl = random.choice(noise_levels)
        methode = random.choice(methods)
        gt, adver = get_image(nl, methode, eval_data)
        true[k, ...] = gt
        adv[k, ...] = adver
    return true, adv
