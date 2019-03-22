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

