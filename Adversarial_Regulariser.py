from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier
from ClassFiles.data_pips import PDB
import platform

if platform.node() == 'motel':
    prefix = '/local/scratch/public/sl767/SPA/'
    pre_data = '/local/scratch/public/sl767/MRC_Data/Data/'
else:
    prefix = 'Data'
    pre_data = ''

TRAINING_PATH = pre_data+'Training/'
EVALUATION_PATH = pre_data+'Evaluation/'
SAVES_PATH = prefix+'Saves/'


class Experiment1(AdversarialRegulariser):
    # experiment name used to determine the Saves folder
    experiment_name = 'LowPassFiltering'

    # Training learning rate
    learning_rate = 0.0001

    # Step Size when solving the variational problem
    step_size = .7
    # The amount of steps of gradient descent performed when evaluating at training time
    total_steps = 20


    def get_network(self):
        return ConvNetClassifier()

    def get_Data_pip(self, training_path, evaluation_path):
        return PDB(training_path, evaluation_path)


experiment = Experiment1(TRAINING_PATH, EVALUATION_PATH, SAVES_PATH)
for k in range(7):
    experiment.train(200)
experiment.log_optimization(30, 1)