import mrcfile
import numpy as np
from ClassFiles.Framework_future import AdversarialRegulariser
from ClassFiles.ut import find
import random
import tensorflow as tf

BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 5
STEPS = 1000
PATH = '/local/scratch/public/sl767/MRC_Data/org/'
PATH_ADV = '/local/scratch/public/sl767/MRC_Data/Data_002_10k/SGD'

train_list = find('*it300_class001.mrc*', PATH_ADV)
train_amount = len(train_list)
print('Training Pictures found: ' + str(train_amount))
eval_list = ''
eval_amount = len(eval_list)
print('Evaluation Pictures found: ' + str(eval_amount))


def locate_gt(adversarial_path):
    pdb_id = adversarial_path[-31:-27]
    l = find('*'+pdb_id+'.mrc', PATH)
    if not len(l)==1:
        raise ValueError('non-unique pdb id: '+l)
    else:
        return l[0]


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
    true = np.zeros(shape=(BATCH_SIZE, 96,96,96))
    adv = np.zeros(shape=(BATCH_SIZE, 96,96,96))
    for k in range(BATCH_SIZE):
        gt, adver = load_data(training_data=training_data)
        true[k, ...] = gt
        adv[k, ...] = adver
    return true, adv

def data_augmentation(gt, adv):
#     eps = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1, 1), minval=0, maxval=1.5)
#     adv_new = tf.multiply(eps, gt) + tf.multiply(tf.ones(shape=(BATCH_SIZE,1,1,1,1))-eps, adv)
    y = tf.spectral.rfft3d(gt[...,0])
    phase = 2*np.pi*tf.random_uniform(shape=tf.shape(y), minval= 0, maxval=1)
    com_phase = tf.exp(1j*tf.cast(phase, tf.complex64))
    y = tf.multiply(com_phase, y)
    return gt, tf.expand_dims(tf.spectral.irfft3d(y), axis=-1)


saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/SGD_Trained/phase_augmentation/'
regularizer = AdversarialRegulariser(saves_path, data_augmentation)


def evaluate():
    gt, adv = get_batch(training_data=True)
    regularizer.test(groundTruth=gt, adversarial=adv, fourier_data =False)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch()
        regularizer.train(groundTruth=gt, adversarial=adv, learning_rate=LEARNING_RATE, fourier_data =False)
        if k%50==0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)
