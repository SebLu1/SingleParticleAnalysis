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
import os

assert len(sys.argv) == 2
star_path = sys.argv[1]


print('Running classical RELION M-step...')
os.environ["RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE"] = "relion_external_reconstruct"
runCommand('relion_external_reconstruct' + ' ' + star_path)
print('Classical RELION M-step Completed')

star_file = load_star(star_path)
target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']
target_path_new = target_path[:-4] + 'POSITIVE' + '.mrc'

os.rename(target_path, target_path_new)

print(target_path + ' renamed to ' + target_path_new )

