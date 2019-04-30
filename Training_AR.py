from ClassFiles.AdversarialRegularizer import AdversarialRegulariser
from ClassFiles.DataAugmentation import interpolation, phase_augmentation, rotation_translation
from ClassFiles.DataGeneration import get_batch

BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 5
STEPS = 1000


def data_augmentation(gt, adv):
    _, adv1 = interpolation(gt, adv)
    _, adv2 = phase_augmentation(gt, adv1)
    new_gt, new_adv = rotation_translation(gt, adv2)
    return new_gt, new_adv


saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/AllData/AllAugmentation/'
regularizer = AdversarialRegulariser(saves_path, data_augmentation)


def evaluate():
    gt, adv = get_batch(eval_data=True, noise_levels=['01', '016'], methods=['EM', 'SGD'])
    regularizer.test(groundTruth=gt, adversarial=adv, fourier_data=False)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch(eval_data=False)
        regularizer.train(groundTruth=gt, adversarial=adv, learning_rate=LEARNING_RATE, fourier_data=False)
        if k%20==0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)
