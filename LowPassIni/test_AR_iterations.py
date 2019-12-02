import numpy as np
import mrcfile
import platform
PLATFORM_NODE = platform.node()
import sys
if PLATFORM_NODE == 'motel':
    sys.path.insert(0, '/home/sl767/PythonCode/SingleParticleAnalysis')
elif PLATFORM_NODE == 'radon':
    sys.path.insert(0, '/home/zickert/SingleParticleAnalysis')
elif 'lmb' in PLATFORM_NODE:
    sys.path.insert(0, '/lmb/home/schools1/SingleParticleAnalysis')
else:
    raise Exception
from ClassFiles.relion_fixed_it import load_star
from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.ut import irfft, rfft
import matplotlib.pyplot as plt
import os
from ClassFiles.ut import cleanStarPath
import itertools
#import subprocess as sp
import argparse
from skimage.measure import compare_ssim as ssim

#%%


if False:
    parser = argparse.ArgumentParser(description='test AR gradient descent')
    parser.add_argument('--gpu', help='GPU to use', required=True)
    parser.add_argument('--positivity', help='AR trained with positivity?', required=True)
    parser.add_argument('--s', help='Sobolev cst', required=True)
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']#'2'
    POS_AR = args['positivity']#'trained_on_non_pos'
    SOBOLEV_CST = args['s']
    PLOT = False
else:
    POS_AR = '1'
    SOBOLEV_CST = '0.5'
    PLOT = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#%%

BASE_PATH = '/mnt/datahd/zickert/'
REGULARIZATION_TIK = 1e-3  # For initialization
PRECOND = False
REPORT = 10
SAVE_RECO_TO_DISK = False
EVAL_METRIC = 'L2_and_SSIM'# 'masked_L2' # 'masked_FSC'
NOISE_LEVEL = '01'
WIN_SIZE = 7 # for SSIM
REGULARIZATION_TIK_GD = 1e6  # During gradient descent


#%%

AR_PATH = '/mnt/datahd/zickert/SPA/Saves/SimDataPaper/Adversarial_Regulariser/no_fancy_data_aug_lr_5e-05_s_{}_pos_{}'.format(SOBOLEV_CST, POS_AR)
regularizer = AdversarialRegulariser(AR_PATH)

POS_AR = ('1' == POS_AR)
    
print('Global parameters:')    
print(SOBOLEV_CST, REGULARIZATION_TIK, PRECOND, REPORT, POS_AR, EVAL_METRIC, NOISE_LEVEL)
print('################################')
#%%
#def runCommand(cmd_string, file_path=None):
#    if file_path is None:
#        sp.call(cmd_string.split(' '))
#    else:
#        file = open(file_path, 'w')
#        sp.call(cmd_string.split(' '), stdout=file)
#        file.close()


#def fscPointFiveCrossing(x, GT_path):
#    raise Exception
#    TEMP_reco_path = BASE_PATH + 'TMP_recos/tmp_reco.mrc'
#    TEMP_reco_masked_path = BASE_PATH + 'TMP_recos/tmp_reco_masked.mrc'
#    TEMP_mask_path = BASE_PATH + 'Test_Learned_Priors/Data/SimDataPaper/Data_001_10k/train/masks/4A2B/mask.mrc'
#    TEMP_mask_log_path = '/beegfs3/zickert/TMP_recos/mask_log.txt'
#    
#    TEMP_FSC_PATH = '/beegfs3/zickert/TMP_recos/tmp_fsc.star'
#
#    with mrcfile.new(TEMP_reco_path, overwrite=True) as mrc:
#        mrc.set_data(x.astype(np.float32))
#        
#    MULT_COMMAND = 'relion_image_handler --i {} --o {} --multiply {}'.format(TEMP_reco_path, TEMP_reco_masked_path, TEMP_mask_path)   
#    runCommand(MULT_COMMAND, TEMP_mask_log_path)
#
#    FSC_COMMAND = 'relion_image_handler --i {} --fsc {} --angpix 1.5'.format(TEMP_reco_masked_path, GT_path)
#    runCommand(FSC_COMMAND, TEMP_FSC_PATH)
# 
#    FSC_dict = load_star(TEMP_FSC_PATH)
#    RES = np.array(FSC_dict['fsc']['rlnAngstromResolution'], dtype='float32')
#    FSC = np.array(FSC_dict['fsc']['rlnFourierShellCorrelation'], dtype='float32')
#
#    cross_idx = np.argmax(FSC <= 0.5)
#    cross_res = RES[cross_idx]
#    return cross_res

#
#def masked_L2(x, GT, PDB):
#    # Make sure Masks exist and paths are correct
#    raise Exception
#    TEMP_reco_path = BASE_PATH + 'TMP_recos/tmp_reco.mrc'
#    TEMP_reco_masked_path = BASE_PATH + 'TMP_recos/tmp_reco_masked.mrc'
#    TEMP_mask_path = BASE_PATH + 'Masks/{p}_mask.mrc'.format(p=PDB)
#    TEMP_mask_log_path = BASE_PATH + 'TMP_recos/mask_log.txt'
#    
#    with mrcfile.new(TEMP_reco_path, overwrite=True) as mrc:
#        mrc.set_data(x.astype(np.float32))
#        
#    MULT_COMMAND = 'relion_image_handler --i {} --o {} --multiply {}'.format(TEMP_reco_path, TEMP_reco_masked_path, TEMP_mask_path)   
#    runCommand(MULT_COMMAND, TEMP_mask_log_path)
#
#    with mrcfile.open(TEMP_reco_masked_path) as mrc:
#        x_masked = mrc.data.copy()
#    
#    return np.sqrt(((x_masked - GT) ** 2).sum())


def L2(x, GT):    
    return np.sqrt(((x - GT) ** 2).sum())

REGULARIZATION_TIK
def vis(data, fourier=True, SCALE=100):
    if fourier:
        data = irfft(data)
    plt.imshow(SCALE*data.squeeze()[..., 45])


def locate_multmap(PDB, noise):
    gt_path = BASE_PATH + 'TrainData/SimDataPaper/Data_0{n}_10k/train/mult_maps/{p}/{p}_mult0{n}.mrc'
    gt_path = gt_path.format(n=noise, p=PDB)
    return gt_path


def locate_star(PDB, noise, iteration):
    star_path =  BASE_PATH + 'TestData/ClassicalRelionData/Data_0{n}_10k/{p}'
    star_path += '/{p}_mult0{n}_it{it}_half1_class001_external_reconstruct.star'
    star_path = star_path.format(p=PDB, n=noise, it=iteration)
    return star_path

#%%
iterable = itertools.product(['4BB9'],#['4MU9', '4AIL', '4BTF', '4A2B', '4BB9', '4M82', '4MU9'],
                             ['001', '005'], ['tik'], ['auto'],#5e4, 1e5],
                             [POS_AR], [100], [1e-3])
for PDB_ID, IT, INI_POINT, AR_REG_TYPE, POSITIVITY, NUM_GRAD_STEPS, STEP_SIZE_NOMINAL in iterable:   

    star_path = locate_star(PDB_ID, NOISE_LEVEL, IT)
    star_file = load_star(star_path)

    gt_path = locate_multmap(PDB_ID, NOISE_LEVEL) 
    with mrcfile.open(gt_path) as mrc:
        gt = mrc.data.copy()

    ### Is this needed?
    real_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsDataReal'])
    imag_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsDataImag'])
    weight_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsWeight'])
    target_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsResult'])
      
    with mrcfile.open(real_data_path) as mrc:
        data_real = mrc.data.copy()
    with mrcfile.open(imag_data_path) as mrc:
        data_im = mrc.data.copy()
    with mrcfile.open(weight_data_path) as mrc:
        kernel = mrc.data.copy()
    with mrcfile.open(target_path) as mrc:
        classical_reco = mrc.data.copy()
       
    complex_data = data_real + 1j * data_im
    
    tikhonov_kernel = kernel + REGULARIZATION_TIK_GD
#    tikhonov_kernel_init = kernel + REGULARIZATION_TIK

       
    if PRECOND:
        precond = np.abs(np.divide(1, tikhonov_kernel))
        precond /= precond.max()
        precond *= 30
    else:
        precond = 1.0

    
    # The scales produce gradients of order 1
    ADVERSARIAL_SCALE = 1 *(96 ** (-0.5))
    DATA_SCALE = 1 / (10 * 96 ** 3)
       
    if INI_POINT == 'tik':
        init = np.divide(complex_data, tikhonov_kernel)
    elif INI_POINT == 'classical':
        init = classical_reco.copy()
        init = rfft(init)
    if POSITIVITY:    
        init = np.fft.rfftn(np.maximum(0, np.fft.irfftn(init)))
    reco = init.copy()

    print('####################')
    print(PDB_ID, NOISE_LEVEL, IT, INI_POINT, AR_REG_TYPE, POSITIVITY, NUM_GRAD_STEPS, STEP_SIZE_NOMINAL)
    print('####################')
#    if EVAL_METRIC == 'masked_FSC':
#        print('INIT FSC 0.5 crossing: ', fscPointFiveCrossing(irfft(init), gt_path))
    if EVAL_METRIC == 'masked_L2':
        print('INIT L2 (masked): ', masked_L2(irfft(init), gt))
    elif EVAL_METRIC == 'L2':
        print('INIT L2: ', L2(irfft(init), gt))
    elif EVAL_METRIC == 'L2_and_SSIM':
        print('INIT L2: ', L2(irfft(init), gt), '. INIT SSIM: ', ssim(irfft(init), gt, win_size=WIN_SIZE))       
    
    if PLOT:
        plt.figure(0, figsize=(10, 3))
        plt.subplot(121)
        vis(gt, fourier=False)
        plt.colorbar()
        plt.subplot(122)
        vis(init)
        plt.colorbar()
        plt.show()
    
    
    if INI_POINT == 'classical' or AR_REG_TYPE != 0 or POSITIVITY:
        # Compute AR Reg param by assuming gt is critical point of objective
        if AR_REG_TYPE == 'auto':
            gradient_gt = regularizer.evaluate(rfft(gt))
            g1_gt = gradient_gt * ADVERSARIAL_SCALE
            g1_gt_norm = np.sqrt((np.abs(g1_gt) ** 2).sum())
            g2_gt = DATA_SCALE * (np.multiply(rfft(gt), tikhonov_kernel) - complex_data)
            g2_gt_norm = np.sqrt((np.abs(g2_gt) ** 2).sum())
            ADVERSARIAL_REGULARIZATION = g2_gt_norm / g1_gt_norm
            print('Auto computed REG_PAR: ', ADVERSARIAL_REGULARIZATION)
        else:
            ADVERSARIAL_REGULARIZATION = AR_REG_TYPE
        
        # Take Gradient steps
        for k in range(NUM_GRAD_STEPS):
            gradient = regularizer.evaluate(reco)
            STEP_SIZE = STEP_SIZE_NOMINAL * 1 / np.sqrt(1 + k / 50)
            g1 = ADVERSARIAL_REGULARIZATION * gradient * ADVERSARIAL_SCALE
            g2 = DATA_SCALE * (np.multiply(reco, tikhonov_kernel) - complex_data)
            
            g = g1 + g2
            reco_o = reco
            reco = reco - STEP_SIZE * precond * g
            
            if POSITIVITY:    
                reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))
        
            if k % REPORT == 0:
                if EVAL_METRIC == 'masked_L2':
                    print('L2 (masked): ', masked_L2(irfft(reco), gt))
                elif EVAL_METRIC == 'L2':
                    print('L2: ', L2(irfft(reco), gt))
                elif EVAL_METRIC == 'L2_and_SSIM':
                    print('L2: ', L2(irfft(reco), gt), '. SSIM: ', ssim(irfft(reco), gt, win_size=WIN_SIZE))  
                if PLOT:
                    plt.figure(k+1, figsize=(20, 3))
                    plt.subplot(151)
                    vis(reco)
                    plt.colorbar()
                    plt.subplot(152)
                    vis(precond * g1)
                    plt.colorbar()
                    plt.subplot(153)
                    vis(precond * g2)
                    plt.colorbar()
                    plt.subplot(154)
                    vis(precond * g)
                    plt.colorbar()
                    plt.subplot(155)
                    vis(reco-init)
                    plt.colorbar()
                    plt.show()

#    if SAVE_RECO_TO_DISK:
#        raise NotImplementedError
#        reco_real = irfft(reco)
#        AR_reco_path = BASE_PATH + 'TMP_recos/reco_it{}_reg{}_pos{}_ini{}_stepLen{}_Nsteps{}.mrc'.format(IT, ADVERSARIAL_REGULARIZATION, POSITIVITY, INI_POINT, STEP_SIZE_NOMINAL, NUM_GRAD_STEPS)  
#        with mrcfile.new(AR_reco_path, overwrite=True) as mrc:
#            mrc.set_data(reco_real.astype(np.float32))
#            mrc.voxel_size = 1.5
