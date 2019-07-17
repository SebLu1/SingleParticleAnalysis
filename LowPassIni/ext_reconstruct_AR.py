#!/home/zickert/miniconda3/envs/tensorflow_gpu/bin/python

import numpy as np
import mrcfile
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
elif PLATFORM_NODE == 'radon':
    sys.path.insert(0, '/home/zickert/SingleParticleAnalysis')
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.ut import irfft, rfft
import os
import subprocess as sp


assert len(sys.argv) == 2
star_path = sys.argv[1]

def runCommand(cmd_string, shell=False):
    sp.call(cmd_string.split(' '), shell=shell)

print('Running classical RELION M-step...')
os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"] = "relion_external_reconstruct"
runCommand('relion_external_reconstruct' + ' ' + star_path)
print('Classical RELION M-step Completed')

star_file = load_star(star_path)
target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']
target_path_TMP = target_path[:-4] + '_TMP_' + '.mrc'

os.rename(target_path, target_path_TMP)

print(target_path + ' renamed to ' + target_path_TMP )
#raise Exception



ADVERSARIAL_REGULARIZATION = float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION"])
print('ADVERSARIAL_REGULARIZATION: ' + str(ADVERSARIAL_REGULARIZATION))

REGULARIZATION_TY = 1e6

if PLATFORM_NODE == 'motel':
    SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/SimDataPaper/'
    SAVES_PATH += 'Adversarial_Regulariser/Cutoff_20/Roto-Translation_Augmentation'
elif PLATFORM_NODE == 'radon':
    SAVES_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/'
    SAVES_PATH += 'Adversarial_Regulariser/trained_on_SGD/Cutoff_20/Roto-Translation_Augmentation'    


iteration = ''
l = star_path.split('_')
for det in l:
    if det[0:2] == 'it':
        iteration = det[2: 5]
        
print('Iteration: {}'.format(iteration))
    
print('Regularization: ' + str(ADVERSARIAL_REGULARIZATION))

star_file = load_star(star_path)

with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data
with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data
with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data

target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']

with mrcfile.open(target_path_TMP) as mrc:
    classical_relion_reco = mrc.data
reco = np.copy(classical_relion_reco)
reco = rfft(reco)
#reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))

complex_data = data_real + 1j * data_im

regularizer = AdversarialRegulariser(SAVES_PATH)

tikhonov_kernel = kernel + REGULARIZATION_TY
precond = np.abs(np.divide(1, tikhonov_kernel))
precond /= precond.max()

# The scales produce gradients of order 1
ADVERSARIAL_SCALE = (96 ** (-0.5))
DATA_SCALE = 1 / (10 * 96 ** 3)

#IMAGING_SCALE=96

for k in range(70):
    STEP_SIZE = 1.0 * 1 / np.sqrt(1 + k / 20)
    
    gradient = regularizer.evaluate(reco)
    g1 = ADVERSARIAL_REGULARIZATION * gradient * ADVERSARIAL_SCALE
#     print(l2(gradient))
    g2 = DATA_SCALE * (np.multiply(reco, tikhonov_kernel) - complex_data)
    
    g = g1 + g2
#     reco = reco - STEP_SIZE * 0.02 * g
    reco = reco - STEP_SIZE * precond * g
    
#    reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))

# write final reconstruction to file
reco_real = irfft(reco)


with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(reco_real.astype(np.float32))
    mrc.voxel_size = 1.5