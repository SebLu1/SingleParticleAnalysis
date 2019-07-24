import mrcfile
import numpy as np
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
elif PLATFORM_NODE == 'lg26.lmb.internal':
    sys.path.insert(0, '/lmb/home/schools1/SingleParticleAnalysis')    
from ClassFiles.ut import getRecos, locate_gt, irfft
from ClassFiles.relion_fixed_it import load_star
import random

DEFAULT_BATCH_SIZE = 1


def get_dict(noise_levels, methodes, eval_data, data_path=None):
    train_dict = {}
    for nl in noise_levels:
        train_dict[nl] = {}
        for met in methodes:
            if met == 'EM' or met == 'def_masked':
                train_dict[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='All', data_path=data_path)
            elif met == 'SGD':
                train_dict[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='Final', data_path=data_path)  
            elif met == 'div':
                train_dict[nl][met] = getRecos(nl, met, eval_data=eval_data, iter='All', data_path=data_path)
            else:
                raise ValueError('Enter valid method')
    return train_dict


def get_image(noise_level, method, data_dict, eval_data):
    data_list = data_dict[noise_level][method]
    adv_path = random.choice(data_list)

    if method == 'div':
#        print('adv_path', adv_path)
#        raise Exception
        star_file = load_star(adv_path)
        with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
            data_real = mrc.data
        with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
            data_im = mrc.data
        with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
            kernel = mrc.data
        adv = np.divide(data_real + 1j * data_im, kernel + 1e-3)
        adv = irfft(adv)
    else:
        with mrcfile.open(adv_path) as mrc:
            adv = mrc.data
    with mrcfile.open(locate_gt(adv_path, eval_data=eval_data)) as mrc:
        gt = mrc.data
#    print(locate_gt(adv_path, eval_data=eval_data))
#    print(star_file)
#    raise Exception
    return gt, adv


def get_batch(noise_levels, methods, batch_size=DEFAULT_BATCH_SIZE,
              data_dict=None, eval_data=False):
    true = np.zeros(shape=(batch_size, 96, 96, 96))
    adv = np.zeros(shape=(batch_size, 96, 96, 96))
    for k in range(batch_size):
        if batch_size != 1:
            # The way on normalizing currently assumes BS = 1
            raise Exception
        nl = random.choice(noise_levels)
        method = random.choice(methods)
        gt, adver = get_image(nl, method, data_dict, eval_data)
        true[k, ...] = gt
        adv[k, ...] = adver
    return true, adv, nl
