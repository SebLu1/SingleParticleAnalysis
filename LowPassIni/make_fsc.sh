make_fsc () {
local MRC_IN="reco_it"$1"_reg"$2"_pos"$3"_ini"$4"_stepLen"$5"_Nsteps"$6".mrc"
local MRC_MASKED="reco_it"$1"_reg"$2"_pos"$3"_ini"$4"_stepLen"$5"_Nsteps"$6"_masked.mrc"
local FSC_STAR="reco_it"$1"_reg"$2"_pos"$3"_ini"$4"_stepLen"$5"_Nsteps"$6"_masked_fsc.star"
local FSC_TXT="reco_it"$1"_reg"$2"_pos"$3"_ini"$4"_stepLen"$5"_Nsteps"$6"_masked_fsc.txt"
relion_image_handler --i $MRC_IN --o $MRC_MASKED --multiply $MASK_PATH
relion_image_handler --i $MRC_MASKED --fsc $GT_PATH --angpix 1.5 > $FSC_STAR
relion_star_printtable $FSC_STAR data_fsc _rlnResolution _rlnFourierShellCorrelation > "$FSC_TXT"
}

IT=$1
INI_POINT=$2

GT_PATH="/ssd/zickert/Data/SimDataPaper/Data_001_10k/train/mult_maps/4A2B/4A2B_mult001.mrc"
MASK_PATH="/beegfs3/zickert/Test_Learned_Priors/Data/SimDataPaper/Data_001_10k/train/masks/4A2B/mask.mrc"
XMGRACE_CMD_POS=""
XMGRACE_CMD=""
if [ $IT = "008" ]; then
#    OLD_FSC="old_fsc_it008.txt"
    OLD_FSC_MASKED="old_fsc_it008.txt"
else
#    OLD_FSC="old_fsc_it001.txt"
    OLD_FSC_MASKED="old_fsc_it001.txt"
fi
StepLen="0.01"
NumSteps="200"


for REG_PAR in 0 40000 # 0 5000 10000 # 20000 30000 40000 50000 # 100000 200000
do 
	for POSITIVITY in False True
	do
		echo "$IT" "$REG_PAR" "$POSITIVITY" "$INI_POINT" "$StepLen" "$NumSteps"
		make_fsc "$IT" "$REG_PAR" "$POSITIVITY" "$INI_POINT" "$StepLen" "$NumSteps"
		if [ $POSITIVITY = "False" ]; then        	
			XMGRACE_CMD="$XMGRACE_CMD reco_it"$IT"_reg"$REG_PAR"_pos"$POSITIVITY"_ini"$INI_POINT"_stepLen"$StepLen"_Nsteps"$NumSteps"_masked_fsc.txt"
		else
			XMGRACE_CMD_POS="$XMGRACE_CMD_POS reco_it"$IT"_reg"$REG_PAR"_pos"$POSITIVITY"_ini"$INI_POINT"_stepLen"$StepLen"_Nsteps"$NumSteps"_masked_fsc.txt"
		fi
	done
done

xmgrace $OLD_FSC_MASKED $XMGRACE_CMD &
xmgrace $OLD_FSC_MASKED $XMGRACE_CMD_POS &
