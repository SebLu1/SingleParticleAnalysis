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


def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass


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
def irfft(fourierData):
    return np.fft.fftshift(np.fft.irfftn(fourierData))

def adjoint_irfft(realData):
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

# TODO: The adjoint of the Fourier transform is taken to be the inverse, which is not exactly correct when using rfft.
class DataTerm(odl.solvers.Functional):
    def __init__(self, domain):
        super(DataTerm, self).__init__(domain)  # , range=odl.RealNumbers())

    def data_gradient(self, prod_elem):

        image = prod_elem[0]
        kernel = prod_elem[1]
        data = prod_elem[2]

        fourier_data = np.fft.fftshift(np.fft.rfftn(image))

        grad = np.multiply(kernel, fourier_data) - data

        return np.fft.ifftshift(np.fft.irfftn(grad))

    def _call(self, prod_elem):
        return l2(self.data_gradient(prod_elem))  # optimal funtional value

    # For performance OrbitLossGradientOperator should maybe defined outside of orbitLoss?
    @property
    def gradient(self):
        class DataTermGrad(odl.Operator):
            def __init__(self, domain, outer_instance):
                super(OrbitLossGradientOperator, self).__init__(domain=domain,
                                                                range=domain)  # , range=odl.RealNumbers())
                self.outer_instance = outer_instance

            def _call(self, prod_elem):
                x = prod_elem[0]  # Ground truth
                y = prod_elem[1]  # Data
                vec_x = np.zeros_like(
                    x)  # Lets set the dervative wrt ground truth-part to zero. It will not propagate back to network params.
                theta = self.outer_instance.localReg(prod_elem)[0]  # optimal angle
                vec_y = y - skimage_rot(x, theta, order=3)
                vec = self.domain.element([vec_x, vec_y])
                return vec

        return OrbitLossGradientOperator(self.domain, self)


