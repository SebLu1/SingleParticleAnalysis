#!/alt/applic/user-maint/sl767/miniconda3/envs/py3/bin/python

import numpy as np
import mrcfile
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.Denoiser import Denoiser
from ClassFiles.ut import irfft

import os
import subprocess as sp 

default_ext_reco_prog = '/home/sl767/bin/relion-devel-lmb/src/apps/external_reconstruct.cpp'

def runCommand(cmd_string):
    sp.call(cmd_string.split(' '))

SAVES_PATH = '/local/scratch/public/sl767/SPA/Saves/Denoiser/All_EM'


assert len(sys.argv) == 2
star_path = sys.argv[1]
star_file = load_star(star_path)

iteration=''
l = star_path.split('_')
for det in l:
    if det[0:2]=='it':
        iteration = det[2:5]
        
print('Iteration: {}'.format(iteration))


target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']

print('Classical Relion M-step...')
os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"] = ""
print('Should be empty: ' + os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"])
runCommand('relion_external_reconstruct' + ' ' + star_path)
print('Classical Relion M-step Completed')
os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"] = "/mhome/maths/s/sl767/PythonCode/SingleParticleAnalysis/External_reco_programs/ext_reconstruct_RED.py"
print('Should be RED ext path: ' + os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"])

den = Denoiser(SAVES_PATH, load=True)

with mrcfile.open(target_path) as mrc:
    rel_reco = mrc.data.copy()

rel_reco_norm = np.sum(np.abs(rel_reco))    

denoised = den.evaluate(rel_reco)
den_norm = np.sum(np.abs(denoised))

#denoised *= rel_reco_norm/den_norm

with mrcfile.new(target_path, overwrite=True) as mrc:
    mrc.set_data(denoised.astype(np.float32))
    mrc.voxel_size = 1.5
