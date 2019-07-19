#!/lmb/home/schools1/anaconda3/envs/tensorflow_gpu/bin/python

import numpy as np
import mrcfile
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
elif PLATFORM_NODE == 'radon':
    sys.path.insert(0, '/home/zickert/SingleParticleAnalysis')
elif 'lmb' in PLATFORM_NODE:
    sys.path.insert(0, '/lmb/home/schools1/SingleParticleAnalysis')
else:
    raise Exception
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import irfft, rfft
import os
import subprocess as sp


assert len(sys.argv) == 2
star_path = sys.argv[1]

def runCommand(cmd_string, shell=False):
    sp.call(cmd_string.split(' '), shell=shell)


INI_POINT = os.environ["RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT"]
REGULARIZATION_TY = float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REG_TIK"])
#ADVERSARIAL_REGULARIZATION = float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION"])
SAVES_PATH = os.environ['RELION_EXTERNAL_RECONSTRUCT_NET_PATH']
NORMALIZATION = os.environ['RELION_EXTERNAL_RECONSTRUCT_NORMALIZATION']

if INI_POINT == 'classical':
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



#print('ADVERSARIAL_REGULARIZATION: ' + str(ADVERSARIAL_REGULARIZATION))




#if PLATFORM_NODE == 'motel':
#    SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/SimDataPaper/'
#    SAVES_PATH += 'Adversarial_Regulariser/Cutoff_20/Roto-Translation_Augmentation'
#elif PLATFORM_NODE == 'radon':
#    SAVES_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/'
#    SAVES_PATH += 'Adversarial_Regulariser/trained_on_SGD/Cutoff_20/Roto-Translation_Augmentation'   


iteration = ''
l = star_path.split('_')
for det in l:
    if det[0:2] == 'it':
        iteration = det[2: 5]
        
print('Iteration: {}'.format(iteration))
    
#print('Regularization: ' + str(ADVERSARIAL_REGULARIZATION))

star_file = load_star(star_path)

with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataReal']) as mrc:
    data_real = mrc.data
with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsDataImag']) as mrc:
    data_im = mrc.data
with mrcfile.open(star_file['external_reconstruct_general']['rlnExtReconsWeight']) as mrc:
    kernel = mrc.data

target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']
complex_data = data_real + 1j * data_im
tikhonov_kernel = kernel + REGULARIZATION_TY


if INI_POINT == 'classical':
    with mrcfile.open(target_path_TMP) as mrc:
        classical_relion_reco = mrc.data
    reco = np.copy(classical_relion_reco)
    reco = rfft(reco)
elif INI_POINT == 'tik':
    reco = np.divide(complex_data, tikhonov_kernel)
    
reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))


denoiser = Denoiser(SAVES_PATH, normalize=NORMALIZATION)

with mrcfile.open(target_path) as mrc:
    reco = mrc.data.copy()

rel_reco_norm = np.sum(np.abs(reco))    

denoised = denoiser.evaluate(reco)
den_norm = np.sum(np.abs(denoised))

#denoised *= rel_reco_norm/den_norm

with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(denoised.astype(np.float32))
    mrc.voxel_size = 1.5
