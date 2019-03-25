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

# Run EM
# In order to do sgd INSTEAD of em, use --sgd
if reconstruct:
    for p in PDB_ID:
        for n in noise_level:
            refine_cmd = 'mpirun -n 3 relion_refine_mpi --o {BP}/Refine3D/{p}/{p}_mult0{n}'
            refine_cmd += ' --auto_refine --split_random_halves --i {BP}/projs/{p}_mult0{n}.star'
            refine_cmd += ' --ref {BP}/SGD/{p}_mult0{n}.mrc --ini_high 30'
            refine_cmd += ' --pad 1'
            refine_cmd += ' --particle_diameter 120 --flatten_solvent --zero_mask --oversampling 1'
            refine_cmd += ' --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5'
            refine_cmd += ' --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale'
            refine_cmd += ' --gpu "{GPU_ids}" --external_reconstruct --maximum_angular_sampling 1.8'
            refine_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
            refine_cmd += ' --pool 30' # Number of images to pool for each thread task
            refine_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network, instead of writing large files to disc
#            refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory. Be careful you have enough RAM for large data sets!
            refine_cmd = refine_cmd.format(BP=base_path, p=p, n=n, GPU_ids=GPU_ids)
            runCommmand(refine_cmd)
