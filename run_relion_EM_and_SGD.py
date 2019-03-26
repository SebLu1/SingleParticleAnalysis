# Script that calls Relion for data generation
import subprocess as sp
import os
import fnmatch

GPU_ids = '2'

mk_dirs = True
create_data = True
SGD = True
EM = False

START_MOL = 0
END_MOL = 50

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

base_path = '/local/scratch/public/sl767/MRC_Data'
train_path = base_path + '/org/training/2'
out_path = base_path + '/Data_002_10k'

PDB_ID = find_PDB_ID('*.mrc', train_path)
PDB_ID = PDB_ID[START_MOL: END_MOL]
#PDB_ID = PDB_ID[:1] # To see that it works



#PDB_ID = ['1H12']
noise_level = ['02']

if mk_dirs:
    for p in PDB_ID:
        runCommand('mkdir -p {OP}/mult_maps/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/projs/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/SGD/{p}'.format(OP=out_path, p=p))
        runCommand('mkdir -p {OP}/EM/{p}'.format(OP=out_path, p=p))
        
if create_data:
    # Scale phantoms
    for p in PDB_ID:
        for n in noise_level:
            mult_cmd = 'relion_image_handler --i {TP}/{p}.mrc --multiply_constant 0.{n} --o {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
            mult_cmd = mult_cmd.format(TP=train_path, OP=out_path, p=p, n=n)
            runCommand(mult_cmd)
    # Create noisy projections
    for p in PDB_ID:
        for n in noise_level:
            proj_cmd = 'relion_project --i {OP}/mult_maps/{p}/{p}_mult0{n}.mrc --o {OP}/projs/{p}/{p}_mult0{n} --nr_uniform 10000 --sigma_offset 2 --add_noise --white_noise 1' 
            proj_cmd = proj_cmd.format(OP=out_path, p=p, n=n)
            runCommand(proj_cmd)

#mkdir -p Refine3D
#mkdir -p Refine3D/${PDB_ID}


# SGD
if SGD:
    for p in PDB_ID:
        for n in noise_level:
            sgd_cmd = 'mpirun -n 3 relion_refine_mpi --o {OP}/SGD/{p}/{p}_mult0{n}'
#            sgd_cmd = 'relion_refine --o {BP}/SGD/{p}/{p}_mult0{n}'
#            sgd_cmd += ' --auto_refine --split_random_halves
            sgd_cmd += ' --i {OP}/projs/{p}/{p}_mult0{n}.star'
            sgd_cmd += ' --denovo_3dref'
#            sgd_cmd += ' --ref {BP}/mult_maps/{p}_mult0{n}.mrc --ini_high 30'
#            sgd_cmd += ' --pad 1'
#            sgd_cmd += ' --particle_diameter 120 --flatten_solvent --zero_mask --oversampling 1'
#            sgd_cmd += ' --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5'
#            sgd_cmd += ' --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale'
            sgd_cmd += ' --gpu "{GPU_ids}"'
#            sgd_cmd += '--external_reconstruct --maximum_angular_sampling 1.8'
            sgd_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
            sgd_cmd += ' --pool 30' # Number of images to pool for each thread task
            sgd_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network, instead of writing large files to disc
            sgd_cmd += ' --sgd'  # Perform stochastic gradient descent instead of default expectation-maximization
            sgd_cmd += ' --sgd_write_iter 50' # : Write out model every so many iterations in SGD (default is writing out all iters)
            sgd_cmd = sgd_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids)
            runCommand(sgd_cmd)

#                --sgd_ini_iter (50) : Number of initial SGD iterations
#                --sgd_fin_iter (50) : Number of final SGD iterations
#         --sgd_inbetween_iter (200) : Number of SGD iterations between the initial and final ones
#               --sgd_ini_resol (35) : Resolution cutoff during the initial SGD iterations (A)
#               --sgd_fin_resol (15) : Resolution cutoff during the final SGD iterations (A)
#             --sgd_ini_subset (100) : Mini-batch size during the initial SGD iterations
#             --sgd_fin_subset (500) : Mini-batch size during the final SGD iterations
 #                        --mu (0.9) : Momentum parameter for SGD updates
 #              --sgd_stepsize (0.5) : Step size parameter for SGD updates
 #     --sgd_sigma2fudge_initial (8) : Initial factor by which the noise variance will be multiplied for SGD (not used if halftime is negative)
#    --sgd_sigma2fudge_halflife (-1) : Initialise SGD with 8x higher noise-variance, and reduce with this half-life in # of particles (default is keep normal variance)
#          --sgd_skip_anneal (false) : By default, multiple references are annealed during the in_between iterations. Use this option to switch annealing off


            #            refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory. Be careful you have enough RAM for large data sets!

    
    
# Run EM
# In order to do sgd INSTEAD of em, use --sgd
if EM:
    for p in PDB_ID:
        for n in noise_level:
            refine_cmd = 'mpirun -n 3 relion_refine_mpi --o {OP}/EM/{p}/{p}_mult0{n}'
            refine_cmd += ' --auto_refine --split_random_halves --i {OP}/projs/{p}/{p}_mult0{n}.star'
            refine_cmd += ' --ref {OP}/SGD/{p}/{p}_mult0{n}.mrc --ini_high 30'
            refine_cmd += ' --pad 1'
            refine_cmd += ' --particle_diameter 120 --flatten_solvent --zero_mask --oversampling 1'
            refine_cmd += ' --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5'
            refine_cmd += ' --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale'
            refine_cmd += ' --gpu "{GPU_ids}" --external_reconstruct --maximum_angular_sampling 1.8'
            refine_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
            refine_cmd += ' --pool 30' # Number of images to pool for each thread task
            refine_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network, instead of writing large files to disc
#            refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory. Be careful you have enough RAM for large data sets!
            refine_cmd = refine_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids)
            runCommand(refine_cmd) 
    
    
