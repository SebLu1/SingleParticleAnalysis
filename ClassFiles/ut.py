import os
import odl
from odl.contrib import tensorflow
import numpy as np
import fnmatch
import tensorflow as tf


def l2(vector):
    return np.sqrt(np.sum(np.square(np.abs(vector))))

# def normalize(vector):
#     if not vector.shape[0] == 96:
#         for k in range(vector.shape[0]):
#             vector[k, ...] = vector[k, ...]/l2(vector[k, ...])
#     else:
#         vector = vector/l2(vector)
#     return vector

def normalize_tf(tensor):
    norms = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1,2,3)))
    norms_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(norms, axis=1), axis=1), axis=1)
    return tf.div(tensor, norms_exp)

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def locate_gt(pdb_id):
    l = find('*'+pdb_id+'.mrc', '/local/scratch/public/sl767/MRC_Data/org/')
    if not len(l)==1:
        raise ValueError('non-unique pdb id: '+l)
    else:
        return l[0]

def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass
        
class Rescaler(object):
    def __init__(self, tensor, batch=True):
        tensor.flags.writeable=True
        self.batch=batch
        self.scales = []
        if batch:
            for k in range(tensor.shape[0]):
                norm = l2(tensor[k,...])
                self.scales.append(norm)
        else:
            norm = l2(tensor)
            self.scales.append(norm)
            
    def normalize(self, tensor):
        if self.batch:
            assert len(self.scales) == tensor.shape[0]
            for k in range(len(self.scales)):
                tensor[k,...] = tensor[k,...] / self.scales[k]
        else:
            tensor = tensor*self.scales[0]       

    def scale_up(self, tensor):
        if self.batch:
            assert len(self.scales) == tensor.shape[0]
            for k in range(len(self.scales)):
                tensor[k,...] = tensor[k,...] * self.scales[k]
        else:
            tensor = tensor*self.scales[0]


def get_coordinate_change(power=1.0, cutoff=100.0):
    print(cutoff)
    print(power)
    X, Y, Z = np.meshgrid(np.linspace(-1, 1, 96),
                          np.linspace(-1, 1, 96),
                          np.linspace(-1, 1, 96))

    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.fftshift(R)[:, :, :49]

    R = 1 / R ** power
    R = np.minimum(R, cutoff * np.min(R))
    R = R / np.max(R)
    return R


IMAGE_SIZE = [96, 96, 96]
FOURIER_SIZE = [96, 96, 49]
space = odl.uniform_discr([0, 0, 0], [1, 1, 1], IMAGE_SIZE, dtype='float32')


class ifftshift_odl(odl.Operator):
    def _call(self, x):
        return space.element(np.fft.ifftshift(x))

    def __init__(self):
        super(ifftshift_odl, self).__init__(space, space, linear=True)


class fftshift_odl(odl.Operator):
    def _call(self, x):
        return space.element(np.fft.fftshift(x))

    def __init__(self):
        super(fftshift_odl, self).__init__(space, space, linear=True)

    @property
    def adjoint(self):
        return ifftshift_odl()

fftshift_tf = odl.contrib.tensorflow.as_tensorflow_layer(fftshift_odl())

# Performs the inverse real fourier transform on Sjors data
SCALING = 96**2
i_SCALING = 1 / SCALING

def irfft(fourierData):
    return SCALING*np.fft.fftshift(np.fft.irfftn(fourierData))

def rfft(realData):
    return i_SCALING*np.fft.rfftn(np.fft.fftshift(realData))

def adjoint_irfft(realData):
    x=FOURIER_SIZE[0]
    y=FOURIER_SIZE[1]
    z=FOURIER_SIZE[2]
    mask = np.concatenate((np.ones(shape=(x,y,1)), 2*np.ones(shape=(x,y,z-2)),np.ones(shape=(x,y,1))), axis=-1)
    fourierData = np.fft.rfftn(np.fft.ifftshift(realData))
    return (np.multiply(fourierData, mask) * SCALING) / (x*y*IMAGE_SIZE[2])

# Ensures consistent Batch,x ,y ,z , channel format
def unify_form(vector):
    n = len(vector.shape)
    if n == 3:
        return np.expand_dims(np.expand_dims(vector, axis =0), axis=-1)
    elif n ==4:
        return np.expand_dims(vector, axis=-1)
    elif n==5:
        return vector
    else:
        raise ValueError('Inputs to the regularizer must have between 3 and 5 dimensions')


