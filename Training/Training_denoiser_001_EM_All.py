import mrcfile
import numpy as np
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import getRecos, locate_gt
import random

saves_path = '/local/scratch/public/sl767/SPA/Saves/Denoiser/All_EM_001_trained_LR_1e-5'


BATCH_SIZE = 1
LEARNING_RATE = 1e-5 #  1e-3 is TF default for Adam
LOOPS = 50
STEPS = 1000

NOISE = '01'
METHOD = 'EM'

train_list = getRecos(noise=NOISE, method=METHOD, iter='All', eval_data=False)
eval_list = getRecos(noise=NOISE, method=METHOD, iter='All', eval_data=True)

train_amount = len(train_list)
print('# Training data points: ' + str(train_amount))
#print(train_list)
eval_amount = len(eval_list)
print('# Evaluation data points: ' + str(eval_amount))
#print(eval_list)
input('Train/Eval Data OK?')
print(saves_path)
input('Saves path OK?')

denoiser = Denoiser(saves_path, solver='GD', load=True)

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




def evaluate():
    gt, adv = get_batch(training_data=True)
    a, b = get_batch(training_data=False)
    denoiser.test(groundTruth=gt, noisy=adv, writer='train')
    denoiser.test(groundTruth=a, noisy=b, writer='test')

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
