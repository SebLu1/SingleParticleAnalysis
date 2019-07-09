#!/alt/applic/user-maint/sl767/miniconda3/envs/py3/bin/python

import numpy as np
import mrcfile
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
from ClassFiles.relion_fixed_it import load_star
#from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.ut import irfft
import os

TIK_REG = float(os.environ["RELION_EXTERNAL_RECONSTRUCTION_TIK_REG"])
print('TIK_REG' + TIK_REG)


#print('-------')
#print('Regularization'+str(ADVERSARIAL_REGULARIZATION))
#print('-------')

REGULARIZATION_TY = TIK_REG
# ADVERSARIAL_REGULARIZATION = 0.0075

#SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/SimDataPaper/'
#SAVES_PATH += 'Adversarial_Regulariser/Cutoff_20/Roto-Translation_Augmentation'

path = sys.argv
assert len(path) == 2
file = load_star(path[1])

iteration=''
l = path[1].split('_')
for det in l:
    if det[0:2]=='it':
        iteration = det[2:5]
        
print('Iteration: {}'.format(iteration))
    
#print('Regularization: '+str(ADVERSARIAL_REGULARIZATION))

with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data

target_path = file['external_reconstruct_general']['rlnExtReconsResult']
complex_data = data_real + 1j * data_im

#regularizer = AdversarialRegulariser(SAVES_PATH)

tikhonov_kernel = kernel + REGULARIZATION_TY
#precondioner = np.abs(np.divide(1, tikhonov_kernel))
#precondioner /= precondioner.max()
tikhonov = np.divide(complex_data, tikhonov_kernel)
reco = np.copy(tikhonov)

#reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))

# The scales produce gradients of order 1
#ADVERSARIAL_SCALE=(96**(-0.5))
#DATA_SCALE=1/(10*96**3)

#IMAGING_SCALE=96

#for k in range(70):
#    STEP_SIZE=1.0 * 1 / np.sqrt(1 + k / 20)
    
#    gradient = regularizer.evaluate(reco)
#    g1 = ADVERSARIAL_REGULARIZATION * gradient * ADVERSARIAL_SCALE
#     print(l2(gradient))
#    g2 = DATA_SCALE*(np.multiply(reco, tikhonov_kernel) - complex_data)
    
#    g = g1 + g2
#     reco = reco - STEP_SIZE * 0.02 * g
#    reco = reco - STEP_SIZE * precondioner * g
    
#    reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))

# write final reconstruction to file
reco_real = irfft(reco)


with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(reco_real.astype(np.float32))
    mrc.voxel_size = 1.5
