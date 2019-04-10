import mrcfile
import collections
from ClassFiles.relion_fixed_it import load_star
import subprocess as sp

def runCommand(cmd_string, file_path=None):
    if file_path is None:
        sp.call(cmd_string.split(' '))
    else:
        file = open(file_path, 'w')
        sp.call(cmd_string.split(' '), stdout=file)
        file.close()


def startingZeros(n):
    if n < 10:
        return '00' + str(n) 
    elif n < 100:
        return '0' + str(n)
    else:
        return str(n)


def continueEM(x, PDB_ID, em_iter_start, em_iter_finish=None, GPU_ids=''):
    if em_iter_finish == None:
        em_iter_finish = em_iter_start + 1
    x = x.squeeze()
    base_path = '/local/scratch/public/sl767/MRC_Data/Data_002_10k'
    
    runCommand('rm -r {BP}/OneStepEM/Input/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    runCommand('rm -r {BP}/OneStepEM/Output/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    runCommand('rm -r {BP}/OneStepEM/FSC/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    
    runCommand('mkdir -p {BP}/OneStepEM/Input/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    runCommand('mkdir -p {BP}/OneStepEM/Output/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    runCommand('mkdir -p {BP}/OneStepEM/FSC/{PDB_ID}'.format(BP=base_path, PDB_ID=PDB_ID))
    
    input_mrc_path = '{BP}/OneStepEM/Input/{PDB_ID}/{PDB_ID}_input.mrc'.format(BP=base_path, PDB_ID=PDB_ID)
    output_mrc_path = '{BP}/OneStepEM/Output/{PDB_ID}/{PDB_ID}_output.mrc'.format(BP=base_path, PDB_ID=PDB_ID)
    projs_star_path = '{BP}/projs/{PDB_ID}/{PDB_ID}_mult002.star'.format(BP=base_path, PDB_ID=PDB_ID)

# Use this if data has been genereated using --split_random_halves        
    input_star_path = '{BP}/EM/{PDB_ID}/{PDB_ID}_mult002_it{EIS}_optimiser.star'.format(BP=base_path, PDB_ID=PDB_ID,
                                                                                        EIS=startingZeros(em_iter_start))

    
    training_path = '/local/scratch/public/sl767/MRC_Data/org/training'
    if PDB_ID[0] == '9':
        training_path = '/local/scratch/public/sl767/MRC_Data/org/eval'
    
    fsc_path = '{BP}/OneStepEM/FSC/{PDB_ID}/{PDB_ID}_fsc.star'.format(BP=base_path, PDB_ID=PDB_ID)
    
    with mrcfile.new(input_mrc_path, overwrite=True) as mrc:
         mrc.set_data(x)
         mrc.voxel_size = 1.5


    ### FSC
    fsc_cmd = 'relion_image_handler --i {IMP}'
    fsc_cmd += ' --fsc {TP}/{PDB_ID_first}/{PDB_ID}.mrc'
#    fsc_cmd += ' --o {FP}' #  Problems with encoding if we write to .star here. Write instead during sp.call()
    fsc_cmd = fsc_cmd.format(IMP=input_mrc_path, TP=training_path,
                             PDB_ID=PDB_ID, PDB_ID_first=PDB_ID[0], FP=fsc_path)
    runCommand(fsc_cmd, fsc_path)
    
    fsc_star_file = load_star(fsc_path)
    fsc = fsc_star_file['fsc']['rlnFourierShellCorrelation']


    ### EM 
    refine_cmd = 'mpirun -n 3 relion_refine_mpi'
    refine_cmd += ' --continue {ISP}'
    refine_cmd += ' --o {OMP}'
    refine_cmd += ' --auto_refine --split_random_halves'
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
    refine_cmd += ' --auto_iter_max {EIF}' #  change this so that it's not hard coded
    refine_cmd = refine_cmd.format(OMP=output_mrc_path[:-4], GPU_ids=GPU_ids, ISP=input_star_path, EIF=em_iter_finish)
    
    runCommand(refine_cmd) 


    ### Accuracy of rotations
    aux_star_path = '{BP}/OneStepEM/Output/{PDB_ID}/{PDB_ID}_output_it{EIS_plus_1}_half1_model.star'
    aux_star_path = aux_star_path.format(BP=base_path, PDB_ID=PDB_ID, EIS_plus_1=startingZeros(em_iter_start + 1))
    #    aux_star_path = '{BP}/OneStepEM/Output/{PDB_ID}/{PDB_ID}_output_it001_model.star'.format(BP=base_path, PDB_ID=PDB_ID)
    aux_star_file = load_star(aux_star_path)
    acc_rot = aux_star_file['model_classes']['rlnAccuracyRotations']
   
    return fsc, acc_rot, output_mrc_path[:-4] + '_it{EIF}_half1_class001.mrc'.format(EIF=startingZeros(em_iter_finish))


if __name__ == '__main__':
    pass