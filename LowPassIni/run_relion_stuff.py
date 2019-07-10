import platform
import argparse
import subprocess as sp
import os
import fnmatch

parser = argparse.ArgumentParser(description='Run RELION stuff')
parser.add_argument('--projs', help='Create projections? (0/1)',
                    required=True)
parser.add_argument('--em', help='Run EM? (0/1)',
                    required=True)
parser.add_argument('--gpu', help='Which GPUs? (empty string means all)',
                    required=True)
parser.add_argument('--eval', help='Which GPUs?',
                    required=True)
parser.add_argument('--pdb_folder', help='PDB-Folder',
                    required=True)
parser.add_argument('--pdb_id', help='PDB ID (if none, give 0)',
                    required=True)
parser.add_argument('--noise', help='Noise levels',
                    required=True)
parser.add_argument('--ext', help='Which exernal reco? (0, def, AR, RED)',
                    required=True)
args = vars(parser.parse_args())

GPU_ids = args['gpu']
NUM_MPI = 3   # At least 3 if --split_random_halves is useds

mk_dirs = True
create_projs = int(args['projs'])
run_EM = int(args['em'])
EVAL_DATA = int(args['eval'])
PDB_FOLDER = args['pdb_folder'].split(' ')
START_MOL = 0
END_MOL = 15

noise_level = args['noise'].split(' ')

EXT_RECO_MODE = args['ext']
if EXT_RECO_MODE == 'def':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = ''
    print('EXT_RECO, should be empty: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])
elif EXT_RECO_MODE == 'AR':
    raise NotImplementedError
elif EXT_RECO_MODE == 'RED':
    raise NotImplementedError

if platform.node() == 'radon':
    BASE_PATH = '/mnt/datahd/zickert/MRC_Data'
MPI_MODE = 'mpirun'


def runCommand(cmd_string, shell=False):
    sp.call(cmd_string.split(' '), shell=shell)


def find_PDB_ID(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append((os.path.join(root, name).replace("\\",
                               "/"))[-8:-4])
    return result


ORG_PATH = BASE_PATH + '/converted'

out_path = BASE_PATH + '/Data/SimDataPaper/Data_0{N}_10k'

if EVAL_DATA:
    out_path = out_path + '/eval'
else:
    out_path = out_path + '/train'

PDB_ID = []
for folder in PDB_FOLDER:
    PDB_ID_tmp = find_PDB_ID('*.mrc', '{OrgP}/{Fol}'.format(OrgP=ORG_PATH,
                             Fol=folder))
    PDB_ID_tmp = PDB_ID_tmp[START_MOL: END_MOL]
    PDB_ID.extend(PDB_ID_tmp)
    print('{OrgP}/{Fol}'.format(OrgP=ORG_PATH,
                             Fol=folder))

if args['pdb_id'] is not '0':
    PDB_ID = args['pdb_id']

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
            refine_cmd += ' --ref {OP}/mult_maps/{p}/{p}_mult0{n}.mrc' 
            refine_cmd += ' --ini_high 30'
            refine_cmd += ' --pad 1'
            refine_cmd += ' --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1'
            refine_cmd += ' --healpix_order 2 --offset_range 5'
            refine_cmd += ' --auto_local_healpix_order 4'
            refine_cmd += ' --offset_step 2 --sym C1'
            refine_cmd += ' --low_resol_join_halves 40'
            refine_cmd += ' --norm --scale'
            refine_cmd += ' --gpu "{GPU_ids}"'
            if EXT_RECO_MODE is not '0':
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
