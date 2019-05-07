from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.DataAugmentation import interpolation, phase_augmentation, rotation_translation, positivity
from ClassFiles.DataGeneration import get_batch

BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 5
STEPS = 5000


# Parameter choices. Heuristic in the BWGAN paper: Choose GAMMA as average dual norm of clean image
# LMB should be bigger than product of norm times dual norm.

# For s=0.0, this implies GAMMA =1.0
# For s=1.0, have GAMMA = 10.0 as realisitc value
S = 1.0
LMB = 10.0
GAMMA = 5.0
CUTOFF = 20.0

def data_augmentation(gt, adv):
    _, adv1 = interpolation(gt, adv)
    _, adv2 = phase_augmentation(gt, adv1)
    _, adv3 = positivity(gt, adv2)
    new_gt, new_adv = rotation_translation(gt, adv3)
    return new_gt, new_adv


saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/Cutoff_20/Translation_Augmentation'
regularizer = AdversarialRegulariser(saves_path, data_augmentation, s=S, cutoff=20.0, lmb=LMB, gamma=GAMMA)


def evaluate():
    gt, adv = get_batch(eval_data=True, noise_levels=['01', '016'], methods=['EM', 'SGD'])
    regularizer.test(groundTruth=gt, adversarial=adv, fourier_data=False)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch(eval_data=False)
        regularizer.train(groundTruth=gt, adversarial=adv, learning_rate=LEARNING_RATE, fourier_data=False)
        if k%50==0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)

LEARNING_RATE = 0.00002

for k in range(LOOPS):
    train(STEPS)

