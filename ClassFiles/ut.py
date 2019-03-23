import os
import odl
from odl.contrib import tensorflow
import numpy as np


def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass


IMAGE_SIZE = [96,96,96]
FOURIER_SIZE = [96,96,49]
space = odl.uniform_discr([0, 0, 0], IMAGE_SIZE, IMAGE_SIZE, dtype='float32')


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
def irfft(fourierData):
    return np.fft.fftshift(np.fft.irfftn(fourierData))

def adjoing_irfft(realData):
    x=FOURIER_SIZE[0]
    y=FOURIER_SIZE[1]
    z=FOURIER_SIZE[2]
    mask = np.concatenate((np.ones(shape=(x,y,1)), 2*np.ones(shape=(x,y,z-2)),np.ones(shape=(x,y,1))), axis=-1)
    fourierData = np.fft.rfftn(np.fft.ifftshift(realData))
    return np.multiply(fourierData, mask)/(x*y*IMAGE_SIZE[2])

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


