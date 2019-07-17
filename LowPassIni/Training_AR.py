import argparse
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
if PLATFORM_NODE == 'radon':
    sys.path.insert(0, '/home/zickert/SingleParticleAnalysis')
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.DataAugmentation import (interpolation, phase_augmentation,
                                        rotation_translation, positivity)
from ClassFiles.DataGeneration import get_batch, get_dict


parser = argparse.ArgumentParser(description='Run RELION stuff')
parser.add_argument('--name', help='Save dir name', required=True)

args = vars(parser.parse_args())

if platform.node() == 'radon':
    BASE_SAVES_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/'
    DATA_PATH = '/mnt/datahd/zickert/MRC_Data/Data/SimDataPaper/'

    
SAVE_NAME = args['name']
SAVES_PATH = BASE_SAVES_PATH + 'Adversarial_Regulariser/{}/Cutoff_20/Roto-Translation_Augmentation'.format(SAVE_NAME)
LOG_FILE_PATH = SAVES_PATH + '/logfile.txt'


BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 2
STEPS = 5000

# Parameter choices. Heuristic in the BWGAN paper:
# Choose GAMMA as average dual norm of clean image
# LMB should be bigger than product of norm times dual norm.

# For s=0.0, this implies GAMMA =1.0
# For s=1.0, have GAMMA = 10.0 as realisitc value
S = 1.0
LMB = 10.0
GAMMA = 5.0
CUTOFF = 20.0

TRAIN_NOISE_LEVELS = ['01']
TRAIN_METHODS = ['EM']
EVAL_NOISE_LEVELS = ['01']
EVAL_METHODS = ['EM']

TRAIN_DICT = get_dict(TRAIN_NOISE_LEVELS, TRAIN_METHODS, eval_data=False,
                      data_path=DATA_PATH)
EVAL_DICT = get_dict(EVAL_NOISE_LEVELS, EVAL_METHODS, eval_data=True,
                     data_path=DATA_PATH)


def data_augmentation(gt, adv):
    _, adv1 = interpolation(gt, adv)
    _, adv2 = phase_augmentation(gt, adv1)
    _, adv3 = positivity(gt, adv2)
    new_gt, new_adv = rotation_translation(gt, adv3)
    return new_gt, new_adv


regularizer = AdversarialRegulariser(SAVES_PATH, data_augmentation,
                                     s=S, cutoff=20.0, lmb=LMB, gamma=GAMMA)

log_file = open(LOG_FILE_PATH, "w")
log_file.write('Train data:\n' + str(TRAIN_DICT) + '\n')
log_file.write('Eval data:\n' + str(EVAL_DICT) + '\n')
log_file.close()



def evaluate():
    gt, adv = get_batch(batch_size=BATCH_SIZE, noise_levels=EVAL_NOISE_LEVELS,
                        methods=EVAL_METHODS, data_dict=EVAL_DICT)
    regularizer.test(groundTruth=gt, adversarial=adv)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch(batch_size=BATCH_SIZE,
                            noise_levels=TRAIN_NOISE_LEVELS,
                            methods=TRAIN_METHODS, data_dict=TRAIN_DICT)
        regularizer.train(groundTruth=gt, adversarial=adv,
                          learning_rate=LEARNING_RATE)
        if k % 50 == 0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)

LEARNING_RATE = 0.00002

for k in range(LOOPS):
    train(STEPS)
