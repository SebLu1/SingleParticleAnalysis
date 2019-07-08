import subprocess as sp
import sys
import os
import platform

arguments = sys.argv

n = sys.argv[1] # Noise level
p = sys.argv[2] # PDB ID
method = sys.argv[3]

if method == 'AR':
    ADVERSARIAL_REGULARIZATION = os.environ["RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION"]#[1:]
    print('Regularization: ' + ADVERSARIAL_REGULARIZATION)

print('Noise Level: ' + str(n))

PLATFORM_NODE = platform.node()

if PLATFORM_NODE == 'motel':
    base_path = '/local/scratch/public/sl767/MRC_Data/Data/SimDataPaper'
    
def runCommand(cmd_string):
    sp.call(cmd_string.split(' '))

out_path = base_path + '/Data_0{}_10k/eval'.format(n)
if method == 'AR':
    out_new_path = base_path + '/Data_0{}_10k/eval/{}/{}/{}'.format(n, method, p, ADVERSARIAL_REGULARIZATION)
else:
    out_new_path = base_path + '/Data_0{}_10k/eval/{}/{}'.format(n, method, p)    

#print(out_new_path)
#raise Exception
    
def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

create_single_folder(out_new_path)

MPI_MODE = 'mpirun'
GPU_ids = ''
NUM_MPI = 3 #  At least 3 if --split_random_halves is used


if MPI_MODE == 'mpirun':
    refine_cmd = 'mpirun -n {NUM_MPI} relion_refine_mpi'
elif MPI_MODE == 'srun':
    refine_cmd = 'srun --mpi=pmi2 relion_refine_mpi'
elif MPI_MODE == 'mpirun-hpc':
    refine_cmd = 'mpirun relion_refine_mpi'
else:
    raise Exception
refine_cmd += ' --o {ONP}/{p}_mult0{n}'
refine_cmd += ' --auto_refine --split_random_halves'
refine_cmd += ' --i {OP}/projs/{p}/{p}_mult0{n}.star'
refine_cmd += ' --ref {OP}/SGD/{p}/{p}_mult0{n}_it300_class001.mrc'# --ini_high 30'
refine_cmd += ' --pad 1'
refine_cmd += ' --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1'
refine_cmd += ' --healpix_order 2 --offset_range 5'
refine_cmd += ' --auto_local_healpix_order 4'
refine_cmd += ' --offset_step 2 --sym C1'
refine_cmd += ' --low_resol_join_halves 40'
refine_cmd += ' --norm --scale'
refine_cmd += ' --gpu "{GPU_ids}"'
refine_cmd += ' --external_reconstruct' # --maximum_angular_sampling 1.8'
refine_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
refine_cmd += ' --pool 30' # Number of images to pool for each thread task
refine_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network,
                                                  # instead of writing large files to disc
#            refine_cmd += ' --iter 30'
#            refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory.
                                   #  Be careful you have enough RAM for large data sets!
refine_cmd = refine_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids, NUM_MPI=NUM_MPI, ONP=out_new_path)
runCommand(refine_cmd)
