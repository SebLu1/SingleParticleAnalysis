# Script that calls Relion for data generation
import subprocess as sp
import os
import fnmatch
import sys
import platform

#SCRIPT_ARGS = sys.argv
#MPI_MODE = SCRIPT_ARGS[1]
#train_folder = SCRIPT_ARGS[2]

#print(SCRIPT_ARGS)

GPU_ids = ''# '0:1' 
NUM_MPI = 3 #  At least 3 if --split_random_halves is used

mk_dirs = input('Make dirs? (True/False):') #False
create_projs = input('Create projections? (True/False):')# False
run_SGD = input('Run SGD? (True/False):')# False
run_EM = input('Run EM? (True/False):')# True
EVAL_DATA = input('Eval data? (True/False):')# True
train_folder = input('PDB folder (0-9)')
START_MOL = int(input('Start-index:')) #0
END_MOL = int(input('Stop-index:')) #30
run_manual_single_PDB = input('Use single PDB instead (thus overriding last three inputs)? (True/False):')# True
if run_manual_single_PDB:
    manual_single_PDB = input('Single PDB id:')# 5A0M

noise_level = input('Noise level(s):').split(' ')

SGD_ini_method = 'lowpass'
SGD_lowpass_frec = 30



PLATFORM_NODE = platform.node()

EXT_RECO = 'REL'


if PLATFORM_NODE == 'motel':
#    print(PLATFORM_NODE)
    base_path = '/local/scratch/public/sl767/MRC_Data'
    MPI_MODE = 'mpirun'
else:
    base_path = '/home/sl767/rds/hpc-work/MRC_Data'
#    MPI_MODE = 'srun'


    
def runCommand(cmd_string, shell=False):
    sp.call(cmd_string.split(' '), shell=shell)

if EXT_RECO == 'REL':
    RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE = '/home/sl767/bin/relion-devel-lmb/src/apps/external_reconstruct.cpp'
elif EXT_RECO == 'AR':
    RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE = '/home/sl767/PythonCode/SingleParticleAnalysis/ext_reconstruct_AR.py'
elif EXT_RECO == 'RED':
    RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE = '/home/sl767/PythonCode/SingleParticleAnalysis/ext_reconstruct_RED.py'

if not EVAL_DATA:
    pass
#    runCommand('export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE={}'.format(RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE), shell=True)

print('RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE:')
input(runCommand('echo $RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE', shell=True))

def find_PDB_ID(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append((os.path.join(root, name).replace("\\", "/"))[-8:-4])
    return result


train_path = base_path + '/org/training'
test_path = base_path + '/org/eval'

out_path = base_path + '/Data/Data_0{N}_10k'

if EVAL_DATA:
    out_path = out_path + '/eval'
else:
    out_path = out_path + '/train'
    
PDB_ID = find_PDB_ID('*.mrc', '{TrP}/{TrF}'.format(TrP=train_path, TrF=train_folder))
PDB_ID = PDB_ID[START_MOL: END_MOL]
if run_manual_single_PDB:
    PDB_ID = [manual_single_PDB]

print('PDB ids: ', PDB_ID)
print('Eval data: ', EVAL_DATA)
input("Looks alright?")

if mk_dirs:
    for p in PDB_ID:
        for n in noise_level:
            runCommand('mkdir -p {OP}/mult_maps/{p}'.format(OP=out_path, p=p).format(N=n))
            runCommand('mkdir -p {OP}/projs/{p}'.format(OP=out_path, p=p).format(N=n))
            runCommand('mkdir -p {OP}/SGD/{p}'.format(OP=out_path, p=p).format(N=n))
            runCommand('mkdir -p {OP}/EM/{p}'.format(OP=out_path, p=p).format(N=n))
            runCommand('mkdir -p {OP}/LowPass/{p}'.format(OP=out_path, p=p).format(N=n))
        
if create_projs:
    # Scale phantoms
    for p in PDB_ID:
        for n in noise_level:
            mult_cmd = 'relion_image_handler --i {TP}/{p1}/{p}.mrc --multiply_constant 0.{n}'
            mult_cmd += ' --o {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            mult_cmd = mult_cmd.format(TP=train_path, OP=out_path, p=p, p1=p[0], n=n)
            runCommand(mult_cmd.format(N=n))
    # Create noisy projections
    for p in PDB_ID:
        for n in noise_level:
            proj_cmd = 'relion_project --i {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            proj_cmd += ' --o {OP}/projs/{p}/{p}_mult0{n} --nr_uniform 10000'
            proj_cmd += ' --sigma_offset 2 --add_noise --white_noise 1' 
            proj_cmd = proj_cmd.format(OP=out_path, p=p, n=n)
            runCommand(proj_cmd.format(N=n))

if SGD_ini_method == 'lowpass' and run_SGD:
    for p in PDB_ID:
        for n in noise_level:
            lp_cmd = 'relion_image_handler --i {TP}/{p1}/{p}.mrc'
            lp_cmd += ' --o {OP}/LowPass/{p}/{p}_lowpass_{SGD_lowpass_frec}_0{n}.mrc' 
            lp_cmd += ' --lowpass {SGD_lowpass_frec} --angpix 1.5'
            lp_cmd += ' --multiply_constant 0.{n}' #  In order to make initial point have correct scaling
            lp_cmd = lp_cmd.format(TP=train_path, OP=out_path,
                                       p=p, p1=p[0], n=n, SGD_lowpass_frec=SGD_lowpass_frec)
            runCommand(lp_cmd.format(N=n))

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
            runCommand(sgd_cmd.format(N=n))

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
            refine_cmd += ' --ref {OP}/SGD/{p}/{p}_mult0{n}_it300_class001.mrc'# --ini_high 30'
            refine_cmd += ' --pad 1'
            refine_cmd += ' --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1'
            refine_cmd += ' --healpix_order 2 --offset_range 5'
            refine_cmd += ' --auto_local_healpix_order 4'
            refine_cmd += ' --offset_step 2 --sym C1'
            refine_cmd += ' --low_resol_join_halves 40'
            refine_cmd += ' --norm --scale'
            refine_cmd += ' --gpu "{GPU_ids}"'
            if not EVAL_DATA:
                pass
            refine_cmd += ' --external_reconstruct'
#--maximum_angular_sampling 1.8'
            refine_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
            refine_cmd += ' --pool 30' # Number of images to pool for each thread task
            refine_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network,
                                                              # instead of writing large files to disc
#            refine_cmd += ' --auto_iter_max 1'    
#            refine_cmd += ' --iter 30'
#            refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory.
                                               #  Be careful you have enough RAM for large data sets!
            refine_cmd = refine_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids, NUM_MPI=NUM_MPI)
            runCommand(refine_cmd.format(N=n)) 
