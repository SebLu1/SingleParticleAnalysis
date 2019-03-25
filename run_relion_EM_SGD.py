# Script that calls Relion for data generation
import subprocess as sp

base_path = '/local/scratch/public/sl767/SPA/playing2_new'
GPU_ids = '2'

create_data = False
reconstruct = True

def runCommmand(cmd_string):
    sp.call(cmd_string.split(' '))


RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE = '?'
#runCommmand('export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE={}'.format(RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE))

PDB_ID = ['1H12']
noise_level = ['02']

#mkdir mult_maps
#mkdir projs

if create_data:
    # Scale phantoms
    for p in PDB_ID:
        for n in noise_level:
            mult_cmd = 'relion_image_handler --i {BP}/{p}.mrc --multiply_constant 0.{n} --o {BP}/mult_maps/{p}_mult0{n}.mrc'
            mult_cmd = mult_cmd.format(BP=base_path, p=p, n=n)
            runCommmand(mult_cmd)
    # Create noisy projections
    for p in PDB_ID:
        for n in noise_level:
            proj_cmd = 'relion_project --i {BP}/mult_maps/{p}_mult0{n}.mrc --o {BP}/projs/{p}_mult0{n} --nr_uniform 10000 --sigma_offset 2 --add_noise --white_noise 1' 
            proj_cmd = proj_cmd.format(BP=base_path, p=p, n=n)
            runCommmand(proj_cmd)

#mkdir -p Refine3D
#mkdir -p Refine3D/${PDB_ID}

# SGD
if reconstruct:
    for p in PDB_ID:
        for n in noise_level:
            sgd_cmd = 'mpirun -n 3 relion_refine_mpi --o {BP}/SGD/{p}/{p}_mult0{n}'
#            sgd_cmd = 'relion_refine --o {BP}/SGD/{p}/{p}_mult0{n}'
#            sgd_cmd += ' --auto_refine --split_random_halves
            sgd_cmd += ' --i {BP}/projs/{p}_mult0{n}.star'
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
            sgd_cmd = sgd_cmd.format(BP=base_path, p=p, n=n, GPU_ids=GPU_ids)
            runCommmand(sgd_cmd)

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

