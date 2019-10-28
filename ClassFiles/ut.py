import os
import odl
from odl.contrib import tensorflow
import numpy as np
import fnmatch
import tensorflow as tf
import platform

PLATFORM_NODE = platform.node()

if PLATFORM_NODE == 'motel':
    BASE_PATH = '/local/scratch/public/sl767/MRC_Data/'
elif PLATFORM_NODE == 'gelfand':
    BASE_PATH = '/mnt/data/zickert/MRC_Data/'
elif PLATFORM_NODE == 'radon':
    BASE_PATH = '/mnt/datahd/zickert/'
    GT_PATH_TRAIN = BASE_PATH + 'TrainData/SimDataPaper/Data_0{N}_10k/train/mult_maps'
    GT_PATH_EVAL = BASE_PATH + 'TrainData/SimDataPaper/Data_0{N}_10k/eval/mult_maps'
elif 'lmb' in PLATFORM_NODE:
    BASE_PATH = '/beegfs3/zickert/PDB2MRC/'
    GT_PATH_TRAIN = '/beegfs3/scheres/PDB2MRC/Data/SimDataPaper/Data_001_10k/train/mult_maps'
    GT_PATH_EVAL = '/beegfs3/scheres/PDB2MRC/Data/SimDataPaper/Data_001_10k/eval/mult_maps'
else:
    raise Exception
DATA_PATH = BASE_PATH + 'TrainData/'
#GT_PATH = BASE_PATH + 'org/'

SSD = False
if SSD:
    GT_PATH_TRAIN = '/ssd/zickert/Data/SimDataPaper/Data_001_10k/train/mult_maps'
    GT_PATH_EVAL = '/ssd/zickert/Data/SimDataPaper/Data_001_10k/eval/mult_maps'


def l2(vector):
    return np.sqrt(np.sum(np.square(np.abs(vector))))

def l2_tf(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.abs(tensor)), axis = (1,2,3,4)))


# def normalize(vector):
#     if not vector.shape[0] == 96:
#         for k in range(vector.shape[0]):
#             vector[k, ...] = vector[k, ...]/l2(vector[k, ...])
#     else:
#         vector = vector/l2(vector)
#     return vector

def startingZeros(n):
    if n < 10:
        return '00' + str(n) 
    elif n < 100:
        return '0' + str(n)
    else:
        return str(n)


def getRecos(noise, method, iter='Final', eval_data=False,
             data_path=DATA_PATH):
    path_list = []
    folder = data_path + 'Data_0{}_10k/'.format(noise)
    if eval_data == True:
        folder += 'eval/'
    else:
        folder += 'train/'
    if method == 'EM':
        folder += 'EM'
        if iter == 'Final':
            path_list = find('*mult0{}_class001.mrc'.format(noise), folder)
        elif iter == 'All':
            path_list = find('*mult0{}_class001.mrc'.format(noise), folder)
            path_list += find('*it*_class001.mrc', folder)
        else:
            path_list = find('*_it{}*_class001.mrc'.format(startingZeros(int(iter))), folder)
    elif method == 'SGD':
        folder += 'SGD'
        if iter == 'Final':
            path_list = find('*it300_class001.mrc', folder)
        elif iter == 'All':
            path_list = find('*it*_class001.mrc', folder)
    elif method == 'def_masked':
        folder += 'def_masked'
        if iter == 'Final':
            path_list = find('*mult0{}_class001.mrc'.format(noise), folder)
        elif iter == 'All':
            path_list = find('*mult0{}_class001.mrc'.format(noise), folder)
            path_list += find('*it*_class001.mrc', folder)
    elif method == 'div':
        folder += 'def_masked'
        if iter == 'Final':
            raise Exception
#            path_list = find('*mult0{}_class001.mrc'.format(noise), folder)
        elif iter == 'All':
            path_list = find('*mult0{}_*_class001_external_reconstruct.star'.format(noise), folder)
#            path_list += find('*it*_class001.mrc', folder) 
#            print('PATH_LIST, in getRecos : ', path_list)
    return path_list


def getStarFiles(noise, method, iter, eval_data=False):
    path_list = []
    folder = DATA_PATH + 'Data_0{}_10k/'.format(noise)
    if eval_data == True:
        folder += 'eval/'
    else:
        folder += 'train/'
    if method == 'EM':
        folder += 'EM'
        if iter == 'Final':
             raise Exception
        elif iter == 'All':
             raise Exception
        else:
            path_list = find('*_it{}*external_reconstruct.star'.format(startingZeros(int(iter))), folder)
    elif method == 'SGD':
        raise Exception
    return path_list


def normalize_tf(tensor):
    norms = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1,2,3)))
    norms_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(norms, axis=1), axis=1), axis=1)
    return tf.div(tensor, norms_exp)


def normalize_np(tensor, return_norm=False):
    norms = np.sqrt(np.sum(np.square(tensor), axis=(1,2,3)))
    norms_exp = np.expand_dims(np.expand_dims(np.expand_dims(norms, axis=1), axis=1), axis=1)
    if return_norm:
        return norms, np.divide(tensor, norms_exp)
    else:
        return np.divide(tensor, norms_exp)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def locate_gt(path, noise, full_path=True, eval_data=False):
    """filename in path should start with pdb_id"""
#    print('path', path)
#    print('full_path', full_path)
#    print('eval_data', eval_data)
    if full_path:
        pdb_id = path.split('/')[-1][:4]
    else:
        pdb_id = path
#    print('pdb_id', pdb_id)
    if eval_data:
        L = find('*' + pdb_id + '_mult0{}.mrc'.format(noise), GT_PATH_EVAL.format(N=noise))
    
    else:
        L = find('*' + pdb_id + '_mult0{}.mrc'.format(noise), GT_PATH_TRAIN.format(N=noise))
#        print(GT_PATH_TRAIN.format(N=noise))
#        raise Exception
    if not len(L) == 1:
        raise ValueError('non-unique pdb id: ' + str(L))
    else:
#        print(L)
#        raise Exception
        return L[0]

def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

class Rescaler(object):
    def __init__(self, tensor, batch=True):
        self.batch = batch
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
                tensor[k,...] /= self.scales[k]
        else:
            tensor *= self.scales[0]       

    def scale_up(self, tensor):
        if self.batch:
            assert len(self.scales) == tensor.shape[0]
            for k in range(len(self.scales)):
                tensor[k,...] *= self.scales[k]
        else:
            tensor *= self.scales[0]

# Cut Fourier masks for negative s
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

# Cut Fourier masks for positive s
def sobolev_mask(power=1.0, cutoff=100.0):    
    print(cutoff)
    print(power)
    X, Y, Z = np.meshgrid(np.linspace(-1, 1, 96),
                          np.linspace(-1, 1, 96),
                          np.linspace(-1, 1, 96))

    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.fftshift(R)

    R = R ** power
    R = np.maximum(R, np.max(R)/cutoff)
    R = R / np.min(R)
    return R


def sobolev_norm(tensor, s=1.0, cutoff=20.0):
    if s == 0.0:
        return l2_tf(tensor)

    else:
        mask = tf.constant(sobolev_mask(power=s, cutoff=cutoff), dtype=tf.complex64)
        sh = tf.shape(tensor)
        scaling = tf.sqrt(tf.cast(sh[1]*sh[2]*sh[3], dtype=tf.complex64))

        # move channels in as fourier transform is taken over the three innermost dimensions
        tensor = tf.transpose(tensor, [0, 4, 1, 2 ,3])
        fourier= tf.spectral.fft3d(tf.cast(tensor, dtype=tf.complex64))/scaling

        masked = tf.multiply(fourier, mask)

        # Use parceval's theorem plus our isometric scaling of the FT here
        return l2_tf(masked)



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
SCALING = 96**2 #  In np one divides the ifft by num_vox^3, in Relion with (num_vox^1 * padding_factor^3)
i_SCALING = 1 / SCALING

def irfft(fourierData, scaling=SCALING):
    return scaling*np.fft.fftshift(np.fft.irfftn(fourierData))

def rfft(realData, i_scaling=i_SCALING):
    return i_scaling*np.fft.rfftn(np.fft.fftshift(realData))

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

def cleanStarPath(x, y):
    x_dir = os.path.split(x)[0]
    y_file = os.path.split(y)[1]
    result = os.path.join(x_dir, y_file)
    return result

