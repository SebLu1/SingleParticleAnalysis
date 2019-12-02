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
parser.add_argument('--pdb_start_idx', help='pdb_start_idx',
                    required=True)
parser.add_argument('--pdb_end_idx', help='pdb_end_idx',
                    required=True)
parser.add_argument('--pdb_id', help='PDB ID (if none, give 0)',
                    required=True)
parser.add_argument('--noise', help='Noise levels',
                    required=True)
parser.add_argument('--ext', help='Which exernal reco? (0, def, def_pos, AR, AR_pos, RED, naive_den)',
                    required=True)
parser.add_argument('--mask', help='Use mask and solvent_correct_fsc?',
                    required=True)
parser.add_argument('--reg_par', help='AR reg. parameter')
#                    ,required=True)
parser.add_argument('--num_mpi', help='MPI nodes',
                    required=True)
parser.add_argument('--gt_start', help='start at GT (no low-pass)')
parser.add_argument('--normalize', help='What normalization was network trained with?')
parser.add_argument('--ini_pt', help='Initial point for AR gradient descent')
parser.add_argument('--tik_reg', help='Tikhononv constant for')
parser.add_argument('--net_path', help='Path to network')
parser.add_argument('--tau_fudge', help='Tau fudge factor')
parser.add_argument('--over_write', help='Run EM even if exists')
parser.add_argument('--sobolev', help='Sobolev constant')


args = vars(parser.parse_args())

print(args)
#raise Exception

if args['tau_fudge'] is not None:
    TAU_FUDGE = float(args['tau_fudge'])
else:
    TAU_FUDGE = 1.0

if args['over_write'] is not None:
    OVER_WRITE = int(args['over_write'])
else:
    OVER_WRITE = 0

if args['gt_start'] is not None:
    GT_START = int(args['gt_start'])
else:
    GT_START = 0

PLATFORM_NODE = platform.node()
if PLATFORM_NODE == 'radon':
    CODE_PATH = '/home/zickert/'
elif 'lmb' in PLATFORM_NODE:
    CODE_PATH = '/lmb/home/schools1/'
else:
    raise Exception      

GPU_ids = args['gpu']
print(GPU_ids)
#os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ids

NUM_MPI = int(args['num_mpi'])   # At least 3 if --split_random_halves is useds

mk_dirs = True
create_projs = int(args['projs'])
run_EM = int(args['em'])
EVAL_DATA = int(args['eval'])
PDB_FOLDER = args['pdb_folder'].split(' ')
START_MOL = int(args['pdb_start_idx'])
END_MOL = int(args['pdb_end_idx'])
MASK = int(args['mask'])
SOBOLEV = args['sobolev']



noise_level = args['noise'].split(' ')
EXT_RECO_MODE = args['ext']
if EXT_RECO_MODE == '0':
    METHOD = 'EM'
else:
    if EXT_RECO_MODE == 'AR' or EXT_RECO_MODE == 'AR_pos':
        if args['reg_par'] is not None:
            REG_PAR = args['reg_par']
        else:
            raise Exception
        if args['ini_pt'] is not None:
            INI_POINT = args['ini_pt']
        else:
            raise Exception
        if args['tik_reg'] is not None:
            TIK_REG = args['tik_reg']
        else:
            raise Exception
        if args['net_path'] is not None:
            NET_PATH = args['net_path']
#            print('NET_PATH' + NET_PATH)
#            raise Exception
        else:
            raise Exception  
        METHOD = EXT_RECO_MODE + '_REG_PAR_' + REG_PAR + '_TIK_REG_' + TIK_REG + '_INI_POINT_' + INI_POINT + '_SOBOLEV_' + SOBOLEV
    elif EXT_RECO_MODE == 'naive_den':
        if args['ini_pt'] is not None:
            INI_POINT = args['ini_pt']
        else:
            raise Exception
        if args['tik_reg'] is not None:
            TIK_REG = args['tik_reg']
        else:
            raise Exception
        if args['net_path'] is not None:
            NET_PATH = args['net_path']
        else:
            raise Exception
        if args['normalize'] is not None:
            NORMALIZE = args['normalize']
        else:
            raise Exception                     
        METHOD = EXT_RECO_MODE + '_TIK_REG_' + TIK_REG + '_INI_POINT_' + INI_POINT + '_NORMALIZE_' + NORMALIZE 

    else:                    
        METHOD = EXT_RECO_MODE
        
if EXT_RECO_MODE == 'def':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = 'relion_external_reconstruct' # Default
    print('EXT_RECO: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])
if EXT_RECO_MODE == 'def_pos':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = CODE_PATH + 'SingleParticleAnalysis/LowPassIni/ext_reconstruct_CLASSICAL_positivity.py' 
    print('EXT_RECO: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])    
elif EXT_RECO_MODE == 'AR':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = CODE_PATH + 'SingleParticleAnalysis/LowPassIni/ext_reconstruct_AR.py'
    print('EXT_RECO: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])
    os.environ['RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION'] = REG_PAR
    os.environ['RELION_EXTERNAL_RECONSTRUCT_REG_TIK'] = TIK_REG
    os.environ['RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT'] = INI_POINT
    os.environ['RELION_EXTERNAL_RECONSTRUCT_NET_PATH'] = NET_PATH  
elif EXT_RECO_MODE == 'AR_pos':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = CODE_PATH + 'SingleParticleAnalysis/LowPassIni/ext_reconstruct_AR_positivity.py'
    print('EXT_RECO: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])
    os.environ['RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION'] = REG_PAR
    os.environ['RELION_EXTERNAL_RECONSTRUCT_REG_TIK'] = TIK_REG
    os.environ['RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT'] = INI_POINT
    os.environ['RELION_EXTERNAL_RECONSTRUCT_NET_PATH'] = NET_PATH    
elif EXT_RECO_MODE == 'RED':
    raise NotImplementedError
elif EXT_RECO_MODE == 'naive_den':
    os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'] = CODE_PATH + 'SingleParticleAnalysis/LowPassIni/ext_reconstruct_naive_den.py'
    print('EXT_RECO: ' + os.environ['RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE'])
#    os.environ['RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION'] = REG_PAR
    os.environ['RELION_EXTERNAL_RECONSTRUCT_REG_TIK'] = TIK_REG
    os.environ['RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT'] = INI_POINT
    os.environ['RELION_EXTERNAL_RECONSTRUCT_NET_PATH'] = NET_PATH    
    os.environ['RELION_EXTERNAL_RECONSTRUCT_NORMALIZATION'] = NORMALIZE                   
                    
#if platform.node() == 'radon':
#    BASE_PATH = '/mnt/datahd/zickert/MRC_Data'
BASE_PATH = '.' #  Now care must be taken w.r.t. where to run 
MPI_MODE = 'mpirun'


def runCommand(cmd_string, shell=False):
    sp.call(cmd_string.split(' '), shell=shell)


def find_PDB_ID(pattern, path, sort=True):
    result_tmp = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result_tmp.append((os.path.join(root, name).replace("\\",
                               "/"))[-8:-4])
    if sort:
        result_tmp.sort()
        result = []
        for j in range(len(result_tmp)):
            if j == 0 or result_tmp[j][:3] != result_tmp[j-1][:3]:
                result.append(result_tmp[j])
        result.sort()   
    else:
        result = result_tmp
    return result


ORG_PATH = BASE_PATH + '/converted'

out_path = BASE_PATH + '/TestAR_Data/SimDataPaper/Data_0{N}_10k'

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
    PDB_ID = [args['pdb_id']]

    
print('PDB ids: ', PDB_ID)
print('Eval data: ', EVAL_DATA)
#input("Looks alright?")
if mk_dirs:
    for p in PDB_ID:
        for n in noise_level:
            runCommand('mkdir -p {OP}/mult_maps/{p}'.format(OP=out_path, p=p).format(N=n))
            runCommand('mkdir -p {OP}/projs/{p}'.format(OP=out_path, p=p).format(N=n))
#            runCommand('mkdir -p {OP}/SGD/{p}'.format(OP=out_path, p=p).format(N=n))
#            runCommand('mkdir -p {OP}/EM/{p}'.format(OP=out_path, p=p).format(N=n))
            if MASK:
                runCommand('mkdir -p {OP}/{meth}_masked/{p}'.format(OP=out_path, p=p, meth=METHOD).format(N=n))
                runCommand('mkdir -p {OP}/masks/{p}'.format(OP=out_path, p=p).format(N=n))
            else:
                runCommand('mkdir -p {OP}/{meth}/{p}'.format(OP=out_path, p=p, meth=METHOD).format(N=n))
#            runCommand('mkdir -p {OP}/LowPass/{p}'.format(OP=out_path, p=p).format(N=n))
       
if create_projs:
    # Scale phantoms
    for p in PDB_ID:
        for n in noise_level:
#            print('{OP}/mult_maps/{p}/{p}_mult0{n}.mrc'.format(OP=out_path, p=p, n=n).format(N=n))
            if not os.path.isfile('{OP}/mult_maps/{p}/{p}_mult0{n}.mrc'.format(OP=out_path, p=p, n=n).format(N=n)):
#                raise Exception
                mult_cmd = 'relion_image_handler --i {OrgP}/{p1}/{p}.mrc --multiply_constant 0.{n}'
                mult_cmd += ' --o {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
                mult_cmd = mult_cmd.format(OrgP=ORG_PATH, OP=out_path, p=p, p1=p[0], n=n)
                runCommand(mult_cmd.format(N=n))
    # Create noisy projections
    for p in PDB_ID:
        for n in noise_level:
#            print('{OP}/projs/{p}/{p}_mult0{n}.mrcs'.format(OP=out_path, p=p, n=n).format(N=n))
            if not os.path.isfile('{OP}/projs/{p}/{p}_mult0{n}.mrcs'.format(OP=out_path, p=p, n=n).format(N=n)):
#                raise Exception
                proj_cmd = 'relion_project --i {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
                proj_cmd += ' --o {OP}/projs/{p}/{p}_mult0{n} --nr_uniform 10000'
                proj_cmd += ' --sigma_offset 2 --add_noise --white_noise 1' 
                proj_cmd = proj_cmd.format(OP=out_path, p=p, n=n)
                runCommand(proj_cmd.format(N=n))

if run_EM:
    for p in PDB_ID:
        for n in noise_level:
            if MASK:
                check_path = '{OP}/{meth}_masked/{p}/{p}_mult0{n}_class001.mrc'.format(OP=out_path, p=p, n=n, meth=METHOD).format(N=n)
            else:
                check_path = '{OP}/{meth}/{p}/{p}_mult0{n}_class001.mrc'.format(OP=out_path, p=p, n=n, meth=METHOD).format(N=n)
#            print(check_path)
            if not os.path.isfile(check_path) or OVER_WRITE:
                print('Running refinement for: ' + p)
#                raise Exception
                if MASK:
                    mask_cmd = 'relion_mask_create --i {OrgP}/{p1}/{p}.mrc'
                    mask_cmd += ' --o {OP}/masks/{p}/mask.mrc'
                    mask_cmd += ' --ini_threshold 0.175 --extend_inimask 0 --width_soft_edge 5 --lowpass 30 --angpix 1.5'
                    mask_cmd = mask_cmd.format(OrgP=ORG_PATH, OP=out_path, p=p, p1=p[0], n=n)
#                    print(mask_cmd)
                    runCommand(mask_cmd.format(N=n))
                if MPI_MODE == 'mpirun':
                    refine_cmd = 'mpirun -n {NUM_MPI} relion_refine_mpi'
                elif MPI_MODE == 'srun':
                    refine_cmd = 'srun --mpi=pmi2 relion_refine_mpi'
                elif MPI_MODE == 'mpirun-hpc':
                    refine_cmd = 'mpirun relion_refine_mpi'
                else:
                    raise Exception 
                if MASK:
                    refine_cmd += ' --solvent_mask {OP}/masks/{p}/mask.mrc --solvent_correct_fsc'
                    refine_cmd += ' --o {OP}/{meth}_masked/{p}/{p}_mult0{n}'
                else:
                    refine_cmd += ' --o {OP}/{meth}/{p}/{p}_mult0{n}'
                refine_cmd += ' --auto_refine --split_random_halves'
                refine_cmd += ' --i {OP}/projs/{p}/{p}_mult0{n}.star'
                refine_cmd += ' --ref {OP}/mult_maps/{p}/{p}_mult0{n}.mrc'
                if not GT_START:
#                    print('good')
#                    raise Exception
                    refine_cmd += ' --ini_high 30'
                else:
                    print('#################################')
                    print('STARTING AT GROUND TRUTH')
                    print('#################################')
                refine_cmd += ' --pad 1'
                refine_cmd += ' --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1'
                refine_cmd += ' --healpix_order 2 --offset_range 5'
                refine_cmd += ' --auto_local_healpix_order 4'
                refine_cmd += ' --offset_step 2 --sym C1'
                refine_cmd += ' --low_resol_join_halves 40'
                refine_cmd += ' --norm --scale'
                refine_cmd += ' --gpu {GPU_ids}'
                if EXT_RECO_MODE is not '0':
                    refine_cmd += ' --external_reconstruct'
    #--maximum_angular_sampling 1.8'
                refine_cmd += ' --j 6' # Number of threads to run in parallel (only useful on multi-core machines)
                refine_cmd += ' --pool 30' # Number of images to pool for each thread task
                refine_cmd += ' --dont_combine_weights_via_disc'  # Send the large arrays of summed weights through the MPI network,
                                                                  # instead of writing large files to disc
    #            refine_cmd += ' --auto_iter_max 1'    
    #            refine_cmd += ' --iter 30'
                refine_cmd += ' --preread_images' #  Use this to let the master process read all particles into memory.
                                                   #  Be careful you have enough RAM for large data sets!
                refine_cmd += ' --tau2_fudge {TAU_F}'
                refine_cmd = refine_cmd.format(OP=out_path, p=p, n=n, GPU_ids=GPU_ids, NUM_MPI=NUM_MPI, meth=METHOD, TAU_F=TAU_FUDGE)
                runCommand(refine_cmd.format(N=n)) 
