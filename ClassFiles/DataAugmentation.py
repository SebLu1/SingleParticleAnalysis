import tensorflow as tf
import numpy as np
from ClassFiles import tensorflow_rotations


INTERPOLATION_INTERVAL = [0, 0.9]


def interpolation(gt, adv):
    batch_size = tf.shape(gt)[0]
    eps = tf.random_uniform(shape=(batch_size, 1, 1, 1, 1), minval=INTERPOLATION_INTERVAL[0],
                            maxval=INTERPOLATION_INTERVAL[1])
    adv_inter = tf.multiply(eps, gt) + tf.multiply(tf.ones(shape=(batch_size,1,1,1,1))-eps, adv)
    return gt, adv_inter


PHASE_SHIFT_INTERVAL = [0, 0.5]


def phase_augmentation(gt, adv):
    batch_size = tf.shape(gt)[0]
    y = tf.spectral.rfft3d(gt[...,0])
    phase = 2*np.pi*tf.random_uniform(shape=tf.shape(y), minval=0, maxval=1)
    com_phase = tf.exp(1j*tf.cast(phase, tf.complex64))
    y = tf.multiply(com_phase, y)
    adv_phase = tf.expand_dims(tf.spectral.irfft3d(y), axis=-1)
    eps1 = tf.random_uniform(shape=(batch_size, 1, 1, 1, 1),
                             minval=PHASE_SHIFT_INTERVAL[0], maxval=PHASE_SHIFT_INTERVAL[1])
    adv_new = tf.multiply(eps1, adv_phase) + tf.multiply(tf.ones(shape=(batch_size,1,1,1,1))-eps1, adv)
    return gt, adv_new


ROTATION_STD = 20
TRANSLATION_MAX = 0.2


def rotation_translation(gt, adv, translation_max = TRANSLATION_MAX):
    batch_size = tf.shape(gt)[0]
    basis_exp = tf.random_normal(shape=[batch_size, 3, 3], stddev=ROTATION_STD)
    skew_exp = basis_exp - tf.transpose(basis_exp, perm=[0, 2, 1])
    rotation = tf.linalg.expm(skew_exp)

    translation = tf.random_uniform(shape=[batch_size, 3, 1], minval=-translation_max, maxval=translation_max)
    theta = tf.concat([rotation, translation], axis=-1)

    rot_gt = tensorflow_rotations.rot3d(gt, theta)
    rot_adv = tensorflow_rotations.rot3d(adv, theta)

    return rot_gt, rot_adv


def positivity(gt, adv):
    return tf.maximum(0.0, gt), tf.maximum(0.0, adv)
