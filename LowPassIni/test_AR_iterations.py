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
from ClassFiles.ut import locate_gt, cleanStarPath
import itertools
import subprocess as sp

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


#%%

#SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/Adversarial_Regulariser/tarball_lr_2e-05_s_1.0_pos_0'
SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/Adversarial_Regulariser/tarball_positive_lr_2e-05_s_1.0_pos_1'
regularizer = AdversarialRegulariser(SAVES_PATH)

#%%


def runCommand(cmd_string, file_path=None):
    if file_path is None:
        sp.call(cmd_string.split(' '))
    else:
        file = open(file_path, 'w')
        sp.call(cmd_string.split(' '), stdout=file)
        file.close()


def fscPointFiveCrossing(x, GT):
    TEMP_reco_path = '/beegfs3/zickert/TMP_recos/tmp_reco.mrc'
    TEMP_reco_masked_path = '/beegfs3/zickert/TMP_recos/tmp_reco_masked.mrc'
    TEMP_mask_path = '/beegfs3/zickert/Test_Learned_Priors/Data/SimDataPaper/Data_001_10k/train/masks/4A2B/mask.mrc'
    TEMP_mask_log_path = '/beegfs3/zickert/TMP_recos/mask_log.txt'


    TEMP_FSC_PATH = '/beegfs3/zickert/TMP_recos/tmp_fsc.star'

    with mrcfile.new(TEMP_reco_path, overwrite=True) as mrc:
        mrc.set_data(x.astype(np.float32))
        
    MULT_COMMAND = 'relion_image_handler --i {} --o {} --multiply {}'.format(TEMP_reco_path, TEMP_reco_masked_path, TEMP_mask_path)   
    runCommand(MULT_COMMAND, TEMP_mask_log_path)

    FSC_COMMAND = 'relion_image_handler --i {} --fsc {} --angpix 1.5'.format(TEMP_reco_masked_path, GT)
    runCommand(FSC_COMMAND, TEMP_FSC_PATH)
 
    FSC_dict = load_star(TEMP_FSC_PATH)
    RES = np.array(FSC_dict['fsc']['rlnAngstromResolution'], dtype='float32')
    FSC = np.array(FSC_dict['fsc']['rlnFourierShellCorrelation'], dtype='float32')

    cross_idx = np.argmax(FSC <= 0.5)
    cross_res = RES[cross_idx]
    return cross_res


#INI_POINT = 'tik'
#INI_POINT = 'classical' #os.environ["RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT"]
REGULARIZATION_TY = 1e-3 #float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REG_TIK"])
#POSITIVITY = False
PRECOND = False
PLOT = False
REPORT = 100
iterable = itertools.product(['001', '008'], ['classical'], [0, 2e4, 5e4, 1e5], [False, True], [1000], [2e-3])
for IT, INI_POINT, ADVERSARIAL_REGULARIZATION, POSITIVITY, NUM_GRAD_STEPS, STEP_SIZE_NOMINAL in iterable:

    #star_path = '/beegfs3/scheres/PDB2MRC/Data/Test/Data_001_10k/4A2B'
    #star_path += '/4A2B_mult001_it008_half1_class001_external_reconstruct.star'
    
    
    star_path = '/beegfs3/zickert/Test_sjors/Data/SimDataPaper/Data_001_10k/4A2B'
    star_path += '/4A2B_mult001_tau16_it{}_half1_class001_external_reconstruct.star'.format(IT)
    
    
    star_file = load_star(star_path)
    
    
    real_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsDataReal'])
    imag_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsDataImag'])
    weight_data_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsWeight'])
    target_path = cleanStarPath(star_path, star_file['external_reconstruct_general']['rlnExtReconsResult'])
    
    
    with mrcfile.open(real_data_path) as mrc:
        data_real = mrc.data
    with mrcfile.open(imag_data_path) as mrc:
        data_im = mrc.data
    with mrcfile.open(weight_data_path) as mrc:
        kernel = mrc.data
    with mrcfile.open(target_path) as mrc:
        classical_reco = mrc.data
    
    
    complex_data = data_real + 1j * data_im
    tikhonov_kernel = kernel + REGULARIZATION_TY
    
    
    
    if PRECOND:
        precond = np.abs(np.divide(1, tikhonov_kernel))
        precond /= precond.max()
    else:
        precond = 1.0
    
    # The scales produce gradients of order 1
    ADVERSARIAL_SCALE = 1 *(96 ** (-0.5))
    DATA_SCALE = 1 / (10 * 96 ** 3)
    
    
    def vis(data, fourier=True, SCALE=100):
        if fourier:
            data = irfft(data)
        plt.imshow(SCALE*data.squeeze()[..., 45])
    #     plt.imshow(np.mean(data.squeeze(), axis=-1))
    
    
    
    
    
    gt_path = locate_gt(star_path)
    with mrcfile.open(gt_path) as mrc:
        gt = mrc.data.copy()
        
    #INI_POINT = 'classical' #os.environ["RELION_EXTERNAL_RECONSTRUCT_AR_INI_POINT"]
    #    ADVERSARIAL_REGULARIZATION = 80000.0 # float(os.environ["RELION_EXTERNAL_RECONSTRUCT_REGULARIZATION"])
    SAVES_PATH = '/beegfs3/zickert/Saves/SimDataPaper/Adversarial_Regulariser/tarball_lr_2e-05_s_1.0_pos_0'

    
    
    if INI_POINT == 'tik':
        init = np.divide(complex_data, tikhonov_kernel)
    elif INI_POINT == 'classical':
        init = classical_reco.copy()
        init = rfft(init)
    if POSITIVITY:    
        init = np.fft.rfftn(np.maximum(0, np.fft.irfftn(init)))
    reco = init.copy()
    print('####################')
    print(IT, INI_POINT, ADVERSARIAL_REGULARIZATION, POSITIVITY, NUM_GRAD_STEPS, STEP_SIZE_NOMINAL)
    print('####################')
    print('INIT FSC 0.5 crossing: ', fscPointFiveCrossing(irfft(init), gt_path))
    
    
    if INI_POINT == 'classical' or ADVERSARIAL_REGULARIZATION != 0 or POSITIVITY:
        for k in range(NUM_GRAD_STEPS):
            STEP_SIZE = STEP_SIZE_NOMINAL * 1 / np.sqrt(1 + k / 50)
            gradient = regularizer.evaluate(reco)
            g1 = ADVERSARIAL_REGULARIZATION * gradient * ADVERSARIAL_SCALE
            g2 = DATA_SCALE * (np.multiply(reco, tikhonov_kernel) - complex_data)
            
            g = g1 + g2
            reco_o = reco
            reco = reco - STEP_SIZE * precond * g
            
            
            if POSITIVITY:    
                reco = np.fft.rfftn(np.maximum(0, np.fft.irfftn(reco)))
        
            if k % REPORT == 0:
                print('FSC 0.5 crossing: ', fscPointFiveCrossing(irfft(reco), gt_path))
                if PLOT:
                    plt.figure(k, figsize=(20, 3))
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
     
    reco_real = irfft(reco)
    AR_reco_path = '/beegfs3/zickert/TMP_recos/reco_it{}_reg{}_pos{}_ini{}_stepLen{}_Nsteps{}.mrc'.format(IT, ADVERSARIAL_REGULARIZATION, POSITIVITY, INI_POINT, STEP_SIZE_NOMINAL, NUM_GRAD_STEPS)
    
    with mrcfile.new(AR_reco_path, overwrite=True) as mrc:
        mrc.set_data(reco_real.astype(np.float32))
        mrc.voxel_size = 1.5