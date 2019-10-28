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
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.DataAugmentation import (interpolation, phase_augmentation,
                                        rotation_translation, positivity)
from ClassFiles.DataGeneration import get_batch, get_dict
from ClassFiles.grid_utils import grid_rot90
from numpy import random


parser = argparse.ArgumentParser(description='Run RELION stuff')
parser.add_argument('--name', help='Save dir name', required=True)
parser.add_argument('--gpu', help='GPU to use', required=True)
parser.add_argument('--lr', help='Learning rate', required=True)
parser.add_argument('--s', help='Sobolev')
parser.add_argument('--train_on_pos', help='Train on positivity projected?')
parser.add_argument('--train_on_div', help='Train on data divided by weight?')


args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

if platform.node() == 'radon':
    BASE_SAVES_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/'
    DATA_PATH = '/mnt/datahd/zickert/TrainData/SimDataPaper/'
elif 'lmb' in PLATFORM_NODE:
    BASE_SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/'
    DATA_PATH = '/beegfs3/scheres/PDB2MRC/Data/SimDataPaper/'    

#SSD = False
#if SSD:
#    DATA_PATH = '/ssd/zickert/Data/SimDataPaper/'

BATCH_SIZE = 1
LEARNING_RATE = float(args['lr'])
LOOPS = 1
STEPS = 5000


if args['train_on_pos'] is not None:
    TRAIN_ON_POS = int(args['train_on_pos'])
else:
    TRAIN_ON_POS = 0

if args['s'] is not None:
    S = float(args['s'])
else:
    S = 1.0
    
if args['train_on_div'] is not None:
    TRAIN_ON_DIVISION = int(args['train_on_div'])
else:
    TRAIN_ON_DIVISION = 0

        
SAVE_NAME = args['name'] + '_lr_' + str(LEARNING_RATE) + '_s_' + str(S) + '_pos_' + str(TRAIN_ON_POS)
SAVES_PATH = BASE_SAVES_PATH + 'Adversarial_Regulariser/{}'.format(SAVE_NAME)
LOG_FILE_PATH = SAVES_PATH + '/logfile.txt'

print('Tensorboard logdir: ' + SAVES_PATH)


LMB = 10.0

# Parameter choices. Heuristic in the BWGAN paper:
# Choose GAMMA as average dual norm of clean image
# LMB should be bigger than product of norm times dual norm.

# For s=0.0, this implies GAMMA =1.0
# For s=1.0, have GAMMA = 10.0 as realisitc value


if 1.0 <= S <= 2.0:
    GAMMA = 5.0
elif S == 0.5:
    GAMMA = 2.5
elif S == 0.0:
    GAMMA = 1.0
else:
    raise Exception

CUTOFF = 20.0

TRAIN_NOISE_LEVELS = ['01', '012', '014']
EVAL_NOISE_LEVELS = ['01', '012', '014']

if TRAIN_ON_DIVISION:
    TRAIN_METHODS = ['div']
    EVAL_METHODS = ['div']
else:
    EVAL_METHODS = ['def_masked']
    TRAIN_METHODS = ['def_masked']

TRAIN_DICT = get_dict(TRAIN_NOISE_LEVELS, TRAIN_METHODS, eval_data=False,
                      data_path=DATA_PATH)
EVAL_DICT = get_dict(EVAL_NOISE_LEVELS, EVAL_METHODS, eval_data=True,
                     data_path=DATA_PATH)



def data_augmentation(gt, adv):#, noise_lvl):
#    _, adv = interpolation(gt, adv)
#    _, adv = phase_augmentation(gt, adv)
    if TRAIN_ON_POS:
#        print('hello')
        _, adv = positivity(gt, adv)
    ### Do 90-rots in numpy instead
#    new_gt, new_adv = rotation_translation(gt, adv)
#    new_gt, new_adv = new_gt/noise_lvl, new_adv/noise_lvl
#    new_gt, new_adv = new_gt/500, new_adv/500 #  Bring back to old scale for training  
    return gt, adv


regularizer = AdversarialRegulariser(SAVES_PATH, data_augmentation,
                                     s=S, cutoff=20.0, lmb=LMB, gamma=GAMMA)

#log_file = open(LOG_FILE_PATH, "w")
#log_file.write('Train data:\n' + str(TRAIN_DICT) + '\n')
#log_file.write('Eval data:\n' + str(EVAL_DICT) + '\n')
#log_file.close()



def evaluate():
    gt, adv, nl = get_batch(batch_size=BATCH_SIZE, noise_levels=EVAL_NOISE_LEVELS,
                        methods=EVAL_METHODS, data_dict=EVAL_DICT,
                        eval_data=True)
#    noise_lvl = int(nl) / 100
    regularizer.test(groundTruth=gt, adversarial=adv)#, noise_lvl=noise_lvl)


def train(steps):
    for k in range(steps):
        gt, adv, nl = get_batch(batch_size=BATCH_SIZE,
                            noise_levels=TRAIN_NOISE_LEVELS,
                            methods=TRAIN_METHODS, data_dict=TRAIN_DICT,
                            eval_data=False)
        rot_id = random.randint(0, 24)

        assert BATCH_SIZE == 1
        gt = gt.squeeze()
        adv = adv.squeeze()
        gt = grid_rot90(gt, rot_id)
        adv = grid_rot90(gt, rot_id)
#        noise_lvl = int(nl) / 100
        regularizer.train(groundTruth=gt, adversarial=adv,
                          learning_rate=LEARNING_RATE)#,
#                          noise_lvl=noise_lvl)
        if k % 50 == 0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)

LEARNING_RATE *= (2/5)

for k in range(LOOPS):
    train(STEPS)
