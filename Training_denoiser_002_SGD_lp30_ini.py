import mrcfile
import numpy as np
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import find, locate_gt
import random

saves_path = '/local/scratch/public/sl767/SPA/Saves/Denoiser/002_SGD_trained_lp30'
denoiser = Denoiser(saves_path, load=True)



BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 5
STEPS = 1000

GT_PATH = '/local/scratch/public/sl767/MRC_Data/org/'

SGD_PATH = '/local/scratch/public/sl767/MRC_Data/Data_002_10k/SGD'
SGD_PATH_MORE_NOISE = '/local/scratch/public/sl767/MRC_Data/Data_001_10k/SGD'
EM_PATH = '/local/scratch/public/sl767/MRC_Data/Data_002_10k/EM'
EM_PATH_MORE_NOISE = '/local/scratch/public/sl767/MRC_Data/Data_001_10k/EM'

train_list_sgd_001 = find('*it300_class001.mrc', SGD_PATH_MORE_NOISE)
train_list_sgd_002 = find('*it300_class001.mrc', SGD_PATH)
train_list_em_001 = find('*mult001_class001.mrc', EM_PATH_MORE_NOISE)
train_list_em_002 = find('*mult002_class001.mrc', EM_PATH)
train_list_em_002_iter001 = find('*it001_half1_class001.mrc', EM_PATH)

train_list = train_list_sgd_002


# Take only particles starting with 3. They have lp30 of gt as ini pt.
tmp = []
for p in train_list:
    if p.split('/')[-1][0] == '3':
        tmp.append(p)
train_list = tmp


train_amount = len(train_list)

print('# Training data points: ' + str(train_amount))

eval_list = ['/local/scratch/public/sl767/MRC_Data/Data_002_10k/SGD/9ICA/9ICA_mult002_it300_class001.mrc']

eval_amount = len(eval_list)
print(eval_list)
print('# Evaluation data points: ' + str(eval_amount))

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
        if k % 5 == 0:
            evaluate()
    denoiser.save()


for k in range(LOOPS):
    train(STEPS)
