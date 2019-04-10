import mrcfile
import numpy as np
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import find
import random


BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 5
STEPS = 1000

GT_PATH = '/local/scratch/public/sl767/MRC_Data/org/'
SGD_PATH = '/local/scratch/public/sl767/MRC_Data/Data_002_10k/SGD'
EM_PATH = '/local/scratch/public/sl767/MRC_Data/Data_002_10k/EM'


NOISY_PATH = EM_PATH

train_list = []
for k in range(1, 9):
    train_list += find('{}*mult002_class001.mrc'.format(k), NOISY_PATH)
train_amount = len(train_list)
print('# Training data points: ' + str(train_amount))
#print(train_list)
eval_list = find('9*mult002_class001.mrc', NOISY_PATH)
eval_amount = len(eval_list)
#print(eval_list)
print('# Evaluation data points: ' + str(eval_amount))


def locate_gt(noisy_path):
    pdb_id = noisy_path[-30: -26]
    L = find('*' + pdb_id + '.mrc', GT_PATH)
    if not len(L) == 1:
        raise ValueError('non-unique pdb id: ' + str(L))
    else:
#        print(pdb_id)
        return L[0]


def get_image(number, training):
    if training:
        L = train_list
    else:
        L = eval_list
    adv_path=L[number]
    with mrcfile.open(adv_path) as mrc:
        adv = mrc.data
    with mrcfile.open(locate_gt(adv_path)) as mrc:
        gt = mrc.data
    return gt, adv


def load_data(training_data=True):
    if training_data:
        n = train_amount
    else:
        n = eval_amount
    return get_image(random.randint(0, n-1), training=training_data)


def get_batch(training_data=True):
    true = np.zeros(shape=(BATCH_SIZE, 96, 96, 96))
    adv = np.zeros(shape=(BATCH_SIZE, 96, 96, 96))
    for k in range(BATCH_SIZE):
        gt, adver = load_data(training_data=training_data)
        true[k, ...] = gt
        adv[k, ...] = adver
    return true, adv


saves_path = '/local/scratch/public/sl767/SPA/Saves/Denoiser/Final_EM_trained'
denoiser = Denoiser(saves_path)


def evaluate():
    gt, adv = get_batch(training_data=True)
    denoiser.test(groundTruth=gt, noisy=adv)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch()
        denoiser.train(groundTruth=gt, noisy=adv,
                       learning_rate=LEARNING_RATE)
        if k % 50 == 0:
            evaluate()
    denoiser.save()


for k in range(LOOPS):
    train(STEPS)
