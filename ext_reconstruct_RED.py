#!/alt/applic/user-maint/sl767/miniconda3/envs/py3/bin/python

import numpy as np
import mrcfile
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import irfft
import sys

REGULARIZATION_TY = 1e6
SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/Densoier/...'

path = sys.argv
assert len(path) == 2
file = load_star(path[1])

iteration = ''
l = path[1].split('_')
for det in l:
    if det[0: 2] == 'it':
        iteration = det[2 : 5]
iteration = int(iteration)
        
        
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data
with mrcfile.open(file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data

target_path = file['external_reconstruct_general']['rlnExtReconsResult']
complex_data = data_real + 1j * data_im


tikhonov_kernel = kernel + REGULARIZATION_TY

tikhonov = np.divide(complex_data, tikhonov_kernel)
reco = np.copy(tikhonov)

denoiser = Denoiser(SAVES_PATH)

def red_reg_grad(x):
    return x - denoiser.evaluate(x)

# The scales produce gradients of order 1
REG_SCALE = (96 ** (-0.5))
DATA_SCALE = 1 / (10 * 96 ** 3)
IMAGING_SCALE = 96
REG_PARAM = 1

for k in range(70):
    STEP_SIZE = 2.0 * 1 / np.sqrt(1 + k / 20)

    g1 = REG_PARAM * REG_SCALE * np.fft.rfft(red_reg_grad(np.fft.irfft(reco)))
    g2 = DATA_SCALE * (np.multiply(reco, tikhonov_kernel) - complex_data)

    g = g1 + g2
    reco = reco + STEP_SIZE * g

    # Enforce positivy
    reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))

# write final reconstruction to file
reco_real = irfft(reco)

print('-------')
print(target_path, file['external_reconstruct_general']['rlnExtReconsResult'])
print('-------')

with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(reco_real.astype(np.float32))
    mrc.voxel_size = 1.5
