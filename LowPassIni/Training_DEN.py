import argparse
import platform
import os
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
elif PLATFORM_NODE == 'radon':
    sys.path.insert(0, '/home/zickert/SingleParticleAnalysis')
elif 'lmb' in PLATFORM_NODE:
    sys.path.insert(0, '/lmb/home/schools1/SingleParticleAnalysis')
else:
    raise Exception    
from ClassFiles.Denoiser import Denoiser
from ClassFiles.DataAugmentation import (interpolation, phase_augmentation,
                                        rotation_translation, positivity)
from ClassFiles.DataGeneration import get_batch, get_dict


parser = argparse.ArgumentParser(description='Run RELION stuff')
parser.add_argument('--name', help='Save dir name', required=True)
parser.add_argument('--gpu', help='GPU to use', required=True)
parser.add_argument('--lr', help='Learning rate', required=True)
parser.add_argument('--s', help='sobolev', required=True)
parser.add_argument('--aug', help='Use phase aug and interpolate?', required=True)

args = vars(parser.parse_args())
sobolev = float(args['s'])
USE_AUG = int(args['aug'])

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

if platform.node() == 'radon':
    BASE_SAVES_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/'
    DATA_PATH = '/mnt/datahd/zickert/MRC_Data/Data/SimDataPaper/'
elif PLATFORM_NODE == 'lg26.lmb.internal' or PLATFORM_NODE == 'pcterm53.lmb.internal':
    BASE_SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/'
    DATA_PATH = '/beegfs3/scheres/PDB2MRC/Data/SimDataPaper/'    
else:
    BASE_SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/'
    DATA_PATH = '/beegfs3/scheres/PDB2MRC/Data/SimDataPaper/'       
    
SAVE_NAME = args['name'] + '_lr_' + args['lr'] + '_s_' + args['s'] + '_aug_' + args['aug']
SAVES_PATH = BASE_SAVES_PATH + 'Denoiser/{}/Roto-Translation_Augmentation'.format(SAVE_NAME)
LOG_FILE_PATH = SAVES_PATH + '/logfile.txt'


BATCH_SIZE = 1
LEARNING_RATE = float(args['lr'])
LOOPS = 50
STEPS = 5000


TRAIN_NOISE_LEVELS = ['01', '012', '014']
TRAIN_METHODS = ['def_masked']
EVAL_NOISE_LEVELS = ['01']
EVAL_METHODS = ['def_masked']

TRAIN_DICT = get_dict(TRAIN_NOISE_LEVELS, TRAIN_METHODS, eval_data=False,
                      data_path=DATA_PATH)
EVAL_DICT = get_dict(EVAL_NOISE_LEVELS, EVAL_METHODS, eval_data=True,
                     data_path=DATA_PATH)



def data_augmentation(gt, adv):#, noise_lvl):
#    print('NOISE LVL', noise_lvl)
#    raise Exception
    if USE_AUG:
        _, adv1 = interpolation(gt, adv)
        _, adv2 = phase_augmentation(gt, adv1)
    #    _, adv3 = positivity(gt, adv2)
        new_gt, new_adv = rotation_translation(gt, adv2)
#        new_gt, new_adv = new_gt/noise_lvl, new_adv/noise_lvl
    else:
#        _, adv1 = interpolation(gt, adv)
#        _, adv2 = phase_augmentation(gt, adv1)
    #    _, adv3 = positivity(gt, adv2)
        new_gt, new_adv = rotation_translation(gt, adv)
#        new_gt, new_adv = new_gt/noise_lvl, new_adv/noise_lvl
#        new_gt, new_adv = new_gt/500, new_adv/500 #  Bring back to old scale for training      
    return new_gt, new_adv


denoiser = Denoiser(SAVES_PATH, data_augmentation, s=sobolev, load=True)

log_file = open(LOG_FILE_PATH, "w")
log_file.write('Train data:\n' + str(TRAIN_DICT) + '\n')
log_file.write('Eval data:\n' + str(EVAL_DICT) + '\n')
log_file.close()



def evaluate():
    gt_e, adv_e, nl_e = get_batch(batch_size=BATCH_SIZE, noise_levels=EVAL_NOISE_LEVELS,
                        methods=EVAL_METHODS, data_dict=EVAL_DICT,
                        eval_data=True)
    gt_t, adv_t, nl_t = get_batch(batch_size=BATCH_SIZE, noise_levels=TRAIN_NOISE_LEVELS,
                        methods=TRAIN_METHODS, data_dict=TRAIN_DICT,
                        eval_data=False)
#    noise_lvl_e = int(nl_e) / 100
#    noise_lvl_t = int(nl_t) / 100
    denoiser.test(groundTruth=gt_e, noisy=adv_e, writer='test')#, noise_lvl=noise_lvl_e)
    denoiser.test(groundTruth=gt_t, noisy=adv_t, writer='train')#, noise_lvl=noise_lvl_t)

def train(steps):
    for k in range(steps):
        gt, adv, nl = get_batch(batch_size=BATCH_SIZE,
                            noise_levels=TRAIN_NOISE_LEVELS,
                            methods=TRAIN_METHODS, data_dict=TRAIN_DICT,
                            eval_data=False)
#        noise_lvl = int(nl) / 100
        denoiser.train(groundTruth=gt, noisy=adv,# noise_lvl=noise_lvl,
                          learning_rate=LEARNING_RATE)
#        print('In train:', noise_lvl)
        if k % 50 == 0:
            evaluate()
    denoiser.save()


for k in range(LOOPS):
    train(STEPS)

#LEARNING_RATE = 0.00001

#for k in range(LOOPS):
#    train(STEPS)
