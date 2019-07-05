#!/alt/applic/user-maint/sl767/miniconda3/envs/py3/bin/python

import numpy as np
import mrcfile
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.ut import irfft
import sys
import os

#ADVERSARIAL_REGULARIZATION_DEFAULT = float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION"])
ADVERSARIAL_REGULARIZATION_DEFAULT = 7e0
TIKHONOV_REGULARIZATION = 2e2
STEP_SIZE_NOMINAL = 1e-4
PADDING_FACTOR = 1 # FIX: int(file['external_reconstruct_general']['rlnPaddingFactor'])
NUM_VOX = 150 #FIX: int(file['external_reconstruct_general']['rlnOriginalImageSize'])
VOX_SIZE = 1.07
TARGET_NUM_VOX = 96
SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/Cutoff_20/Translation_Augmentation'

path = sys.argv
assert len(path) == 2
file = load_star(path[1])

iteration = ''
l = path[1].split('_')
for det in l:
    if det[0:2]=='it':
        iteration = int(det[2:5])
if iteration < 5:
    ADVERSARIAL_REGULARIZATION = ADVERSARIAL_REGULARIZATION_DEFAULT
else:
    ADVERSARIAL_REGULARIZATION = ADVERSARIAL_REGULARIZATION_DEFAULT

print('Iteration: {}'.format(iteration))
    
print('-------')
print('Regularization' + str(ADVERSARIAL_REGULARIZATION))
print('-------')

with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data.copy()
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data.copy()
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data.copy()

target_path = file['external_reconstruct_general']['rlnExtReconsResult']

regularizer = AdversarialRegulariser(SAVES_PATH)

complex_data = data_real + 1j * data_im

complex_data_norm = np.mean(irfft(complex_data, scaling=NUM_VOX**2))
complex_data /= complex_data_norm
kernel /= complex_data_norm
tikhonov_kernel = kernel + TIKHONOV_REGULARIZATION

#precond = np.abs(np.divide(1, tikhonov_kernel))
#precond /= precond.max()
precond = 1
tikhonov = np.divide(complex_data, tikhonov_kernel)
reco = np.copy(tikhonov)

for k in range(150):
    STEP_SIZE = STEP_SIZE_NOMINAL / np.sqrt(1 + k / 20)
    
    ###############
    # DOWNSAMPLING
    reco_ds = np.copy(reco)
    reco_ds = np.fft.fftshift(reco_ds, axes=(0,1))
    reco_ds = reco_ds[NUM_VOX//2-TARGET_NUM_VOX//2 : NUM_VOX//2+TARGET_NUM_VOX//2, 
                      NUM_VOX//2-TARGET_NUM_VOX//2 : NUM_VOX//2+TARGET_NUM_VOX//2, 
                      0:(TARGET_NUM_VOX//2)+1]
    reco_ds = np.fft.ifftshift(reco_ds, axes=(0,1))
    ###############
    
    gradient_tmp = regularizer.evaluate(reco_ds)
    
    ###############
    # UPSAMPLING
    gradient = np.zeros_like(complex_data)
    gradient[NUM_VOX//2-TARGET_NUM_VOX//2 : NUM_VOX//2+TARGET_NUM_VOX//2, 
             NUM_VOX//2-TARGET_NUM_VOX//2 : NUM_VOX//2+TARGET_NUM_VOX//2, 
             0:(TARGET_NUM_VOX//2)+1] = np.fft.fftshift(gradient_tmp, axes=(0,1))
    gradient = np.fft.ifftshift(gradient, axes=(0,1))
    ###############
    
    g1 = ADVERSARIAL_REGULARIZATION * gradient
    g2 = (np.multiply(reco, tikhonov_kernel) - complex_data)
    
    g = g1 + g2
    reco = reco - STEP_SIZE * precond * g
    reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))
    
# write final reconstruction to file
reco_real = irfft(reco, scaling=((1/PADDING_FACTOR ** 3) * NUM_VOX ** 2))


with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(reco_real.astype(np.float32))
    mrc.voxel_size = VOX_SIZE
