# Script that calls Relion for data generation
import subprocess as sp
import os
import fnmatch
import sys
import platform

SCRIPT_ARGS = sys.argv
MPI_MODE = SCRIPT_ARGS[1]
train_folder = SCRIPT_ARGS[2]

#print(SCRIPT_ARGS)

GPU_ids = ''# '0:1' 
NUM_MPI = 2 #  At least 3 if --split_random_halves is used

mk_dirs = True
create_projs = True
run_SGD = True
run_EM = True

SGD_ini_method = 'lowpass'
SGD_lowpass_frec = 30

START_MOL = 0
END_MOL = 30

PLATFORM_NODE = platform.node()

#MPI_MODE = None

if PLATFORM_NODE == 'motel':
    print(PLATFORM_NODE)
    base_path = '/local/scratch/public/sl767/MRC_Data'
#    MPI_MODE = 'mpirun'
else:
    base_path = '/home/sl767/rds/hpc-work/MRC_Data'
#    MPI_MODE = 'srun'


    
def runCommand(cmd_string):
    sp.call(cmd_string.split(' '))


RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE = '?'
#runCommmand('export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE={}'.format(RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE))

def find_PDB_ID(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append((os.path.join(root, name).replace("\\", "/"))[-8:-4])
    return result


train_path = base_path + '/org/training'
test_path = base_path + '/org/eval'

#train_path = test_path #  Hack for now
noise_level = ['02'] #  Right now this has to be a list with a single element
out_path = base_path + '/Data_0{}_10k'.format(noise_level[0])

PDB_ID = find_PDB_ID('*.mrc', '{TrP}/{TrF}'.format(TrP=train_path, TrF=train_folder))
#PDB_ID = find_PDB_ID('*.mrc', '{TP}/9'.format(TP=train_path))
PDB_ID = PDB_ID[START_MOL: END_MOL]
#PDB_ID = PDB_ID[:1] # To see that it works
#PDB_ID = ['3PE7']

if len(SCRIPT_ARGS) == 4:
    PDB_ID = [SCRIPT_ARGS[3]] 


if mk_dirs:
    for p in PDB_ID:
        runCommand('mkdir -p {OP}/mult_maps/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/projs/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/SGD/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/EM/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/LowPass/{p}'.format(OP=out_path, p=p))
        
if create_projs:
    # Scale phantoms
    for p in PDB_ID:
        for n in noise_level:
            mult_cmd = 'relion_image_handler --i {TP}/{p1}/{p}.mrc --multiply_constant 0.{n}'
            mult_cmd += ' --o {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            mult_cmd = mult_cmd.format(TP=train_path, OP=out_path, p=p, p1=p[0], n=n)
            runCommand(mult_cmd)
    # Create noisy projections
    for p in PDB_ID:
        for n in noise_level:
            proj_cmd = 'relion_project --i {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            proj_cmd += ' --o {OP}/projs/{p}/{p}_mult0{n} --nr_uniform 10000'
            proj_cmd += ' --sigma_offset 2 --add_noise --white_noise 1' 
            proj_cmd = proj_cmd.format(OP=out_path, p=p, n=n)
            runCommand(proj_cmd)

if SGD_ini_method == 'lowpass':
    for p in PDB_ID:
        for n in noise_level:
            lp_cmd = 'relion_image_handler --i {TP}/{p1}/{p}.mrc'
            lp_cmd += ' --o {OP}/LowPass/{p}/{p}_lowpass_{SGD_lowpass_frec}_0{n}.mrc' 
            lp_cmd += ' --lowpass {SGD_lowpass_frec} --angpix 1.5'
            lp_cmd += ' --multiply_constant 0.{n}' #  In order to make initial point have correct scaling
            lp_cmd = lp_cmd.format(TP=train_path, OP=out_path,
                                       p=p, p1=p[0], n=n, SGD_lowpass_frec=SGD_lowpass_frec)
            runCommand(lp_cmd)

if run_SGD:
    for p in PDB_ID:
        for n in noise_level:
            if MPI_MODE == 'mpirun':
                sgd_cmd = 'mpirun -n {NUM_MPI} relion_refine_mpi'
            elif MPI_MODE == 'srun':
                sgd_cmd = 'srun --mpi=pmi2 relion_refine_mpi'
            elif MPI_MODE == 'mpirun-hpc':
                sgd_cmd = 'mpirun relion_refine_mpi'
            else:
                raise Exception 
            sgd_cmd += ' --o {OP}/SGD/{p}/{p}_mult0{n}'
            sgd_cmd += ' --i {OP}/projs/{p}/{p}_mult0{n}.star'
            if SGD_ini_method == 'lowpass':
                sgd_cmd += ' --ref {OP}/LowPass/{p}/{p}_lowpass_{SGD_lowpass_frec}_0{n}.mrc'  
            elif SGD_ini_method == 'denovo':
                sgd_cmd += ' --denovo_3dref'
            elif SGD_ini_method == 'cheat':
                sgd_cmd += ' --ref {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            else: 
                raise Exception
            sgd_cmd += ' --ini_high 30'
            sgd_cmd += ' --gpu "{GPU_ids}"'
            sgd_cmd += ' --pad 1'
            sgd_cmd += ' --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1'
            sgd_cmd += ' --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5'
            sgd_cmd += ' --offset_step 2 --sym C1'
#            sgd_cmd += ' --auto_refine --split_random_halves #  Should we do this also for SGD?
#            sgd_cmd += ' --low_resol_join_halves 40'         #  Should we do this also for SGD?
            sgd_cmd += ' --norm --scale'
            sgd_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
            sgd_cmd += ' --pool 30' # Number of images to pool for each thread task
            sgd_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network,
                                                           # instead of writing large files to disc
            sgd_cmd += ' --sgd'  # Perform stochastic gradient descent instead of default expectation-maximization
            sgd_cmd += ' --sgd_write_iter 50' # : Write out model every so many iterations in SGD (default is writing out all iters)
            sgd_cmd = sgd_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids, SGD_lowpass_frec=SGD_lowpass_frec, NUM_MPI=NUM_MPI)
            runCommand(sgd_cmd)

if run_EM:
    for p in PDB_ID:
        for n in noise_level:
            if MPI_MODE == 'mpirun':
                refine_cmd = 'mpirun -n {NUM_MPI} relion_refine_mpi'
            elif MPI_MODE == 'srun':
                refine_cmd = 'srun --mpi=pmi2 relion_refine_mpi'
            elif MPI_MODE == 'mpirun-hpc':
                refine_cmd = 'mpirun relion_refine_mpi'
            else:
                raise Exception 
            refine_cmd += ' --o {OP}/EM/{p}/{p}_mult0{n}'
            refine_cmd += ' --auto_refine --split_random_halves'
            refine_cmd += ' --i {OP}/projs/{p}/{p}_mult0{n}.star'
            refine_cmd += ' --ref {OP}/SGD/{p}/{p}_mult0{n}_it300_class001.mrc --ini_high 30'
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
            refine_cmd = refine_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids, NUM_MPI=NUM_MPI)
            runCommand(refine_cmd) 
