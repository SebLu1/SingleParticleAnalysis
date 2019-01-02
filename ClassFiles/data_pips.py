import fnmatch
import os
from abc import ABC, abstractmethod
import mrcfile
import random

# some handy methods
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def replace_extension(path, old_ex, new_ex):
    l = len(old_ex)
    pruned_path = path[0:-l]
    path = pruned_path+new_ex
    return path

def get_paths(path, truth_ex, recon_ex):
    L = find('*'+truth_ex, path)
    res = []
    for l in L:
        res.append([l, replace_extension(l, truth_ex, recon_ex)])
    return res


# Abstract class for data preprocessing. To customize to your own dataset, define subclass with the
# image_size, name and color of your dataset and the corresponding load_data method
class data_pip(ABC):
    image_size = [64, 64, 64]
    name = 'PDB'

    # load data outputs single image in format (image_size, colors).
    # The image should be normalized between (0,1).
    # The training_data flag determines if the image should be taken from training or test set
    @abstractmethod
    def load_data(self, training_data=True):
        pass

# returns 128x128 image from the BSDS dataset.
class PDB(data_pip):

    def __init__(self, training_path, evaluation_path):
        super(PDB, self).__init__()
        # set up the training data file system
        print(training_path)
        self.train_list = get_paths(training_path, '_scaled.mrc', '_filtered.mrc')
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        print(evaluation_path)
        self.eval_list = get_paths(evaluation_path, '_scaled.mrc', '_filtered.mrc')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))

    def get_image(self, number, training):
        if training:
            L = self.train_list
        else:
            L = self.eval_list
        with mrcfile.open(L[number][0]) as mrc:
            gt = mrc.data
        with mrcfile.open(L[number][1]) as mrc:
            rec = mrc.data
        return gt, rec

    # methode to cut a image_size area out of the training images
    def load_data(self, training_data=True):
        if training_data:
            n = self.train_amount
        else:
            n = self.eval_amount
        return self.get_image(random.randint(n), training=training_data)

