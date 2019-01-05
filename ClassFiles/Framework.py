import numpy as np
import os
from abc import ABC, abstractmethod
import tensorflow as tf
import mrcfile


class GenericFramework(ABC):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    @abstractmethod
    def get_network(self):
        # returns an object of the network class. Used to set the network used
        pass

    @abstractmethod
    def get_Data_pip(self, training_path, evaluation_path):
        # returns an object of the data_pip class.
        pass

    @staticmethod
    def create_single_folder(folder):
        # creates folder and catches error if it exists already
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass

    def __init__(self, training_path, evaluation_path, saves_path):
        self.data_pip = self.get_Data_pip(training_path, evaluation_path)
        self.image_size = self.data_pip.image_size
        self.network = self.get_network()

        # finding the correct path for saving models
        self.path = saves_path+'Saves/{}/{}/{}/'.format(self.data_pip.name, self.model_name, self.experiment_name)
        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        self.create_single_folder(self.path+'Data')
        self.create_single_folder(self.path + 'Logs')

    def generate_training_data(self, batch_size, training_data=True):
        # method to generate training data given the current model type
        true = np.empty([batch_size] + self.image_size + [1], dtype='float32')
        estimate = np.empty([batch_size]+ self.image_size + [1], dtype='float32')
        for k in range(batch_size):
            gt, rec = self.data_pip.load_data(training_data=training_data)
            true[k,...,0]=gt
            estimate[k,...,0]=rec
        return true, estimate

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    @abstractmethod
    def evaluate(self, image):
        # apply the model to data
        pass


class AdversarialRegulariser(GenericFramework):
    model_name = 'Adversarial_Regulariser'
    # the absolut noise level
    batch_size = 16
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0001
    # default step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps = 20

    # sets up the network architecture
    def __init__(self, training_path, evaluation_path, saves_path):
        # call superclass init
        super(AdversarialRegulariser, self).__init__(training_path, evaluation_path, saves_path)

        ### Training the regulariser ###

        # placeholders for NN
        self.gen = tf.placeholder(shape=[None] + self.image_size + [1], dtype=tf.float32)
        self.true = tf.placeholder(shape=[None]+self.image_size + [1], dtype=tf.float32)

        # the network outputs
        self.gen_was = self.network.net(self.gen)
        self.data_was = self.network.net(self.true)

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)

        # intermediate point
        random_int = tf.random_uniform([tf.shape(self.true)[0], 1, 1, 1, 1], 0.0, 1.0)
        self.inter = tf.multiply(self.gen, random_int) + tf.multiply(self.true, 1 - random_int)
        self.inter_was = tf.reduce_sum(self.network.net(self.inter))

        # calculate derivative at intermediate point
        self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

        # take the L2 norm of that derivative
        self.norm_gradient = tf.sqrt(tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3)))
        self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(self.norm_gradient - 1)))

        # Overall Net Training loss
        self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        ### The reconstruction network ###

        # placeholders
        self.reconstruction = tf.placeholder(shape=[None] + self.image_size + [1], dtype=tf.float32)
        self.ground_truth = tf.placeholder(shape=[None] + self.image_size + [1], dtype=tf.float32)

        # the loss functional
        self.was_output = tf.reduce_mean(self.network.net(self.reconstruction))
        self.was_cor = self.was_output - tf.reduce_mean(self.network.net(self.ground_truth))

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already. Makes gradients scaling batch size inveriant
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.was_output * batch_s, self.reconstruction)

        # Measure quality of reconstruction
        self.cut_reco = tf.clip_by_value(self.reconstruction, 0.0, 1.0)
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.ground_truth - self.reconstruction),
                                                            axis=(1, 2, 3))))

        # logging tools
        with tf.name_scope('Network_Optimization'):
            dd = tf.summary.scalar('Data_Difference', self.wasserstein_loss)
            lr = tf.summary.scalar('Lipschitz_Regulariser', self.regulariser_was)
            ol = tf.summary.scalar('Overall_Net_Loss', self.loss_was)
            self.merged_network = tf.summary.merge([dd, lr, ol])

        sliceN = tf.cast((tf.shape(self.ground_truth)[3]/2), dtype=tf.int32)
        with tf.name_scope('Picture_Optimization'):
            wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_cor)
            recon = tf.summary.image('Reconstruction', self.cut_reco[..., sliceN, :], max_outputs=1)
            ground_truth = tf.summary.image('Ground_truth', self.ground_truth[..., sliceN, :], max_outputs=1)
            quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
            self.merged_pic = tf.summary.merge([wasser_loss, quality_assesment, recon, ground_truth])

        # set up the logger
        self.writer = tf.summary.FileWriter(self.path + 'Logs/Network_Optimization/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def update_pic(self, steps, stepsize, guess):
        # updates the guess to come closer to the solution of the variational problem.
        for k in range(steps):
            gradient = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: guess})
            guess = guess - stepsize * gradient[0]
        return guess

    def log_network_training(self):
        true, estimate = self.generate_training_data(batch_size=self.batch_size, training_data=False)
        logs, step = self.sess.run([self.merged_network, self.global_step],
                                   feed_dict={self.gen: estimate, self.true: true})
        self.writer.add_summary(logs, step)

    def log_optimization(self, steps=None, step_s=None):
        if steps is None:
            steps = self.total_steps
        if step_s is None:
            step_s = self.step_size

        true, estimate = self.generate_training_data(32, training_data=False)
        guess = np.copy(estimate)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/step_s_{}'.format(step_s))
        for k in range(steps+1):
            summary = self.sess.run(self.merged_pic,
                                    feed_dict={self.reconstruction: guess,
                                               self.ground_truth: true})
            writer.add_summary(summary, k)
            guess = self.update_pic(1, step_s, guess)
        writer.flush()
        writer.close()

    def visualize_optimization(self, steps, step_s):
        true, estimate = self.generate_training_data(1, training_data=False)
        guess = np.copy(estimate)
        path = self.path + 'Images/step_s_{}_steps_{}'.format(step_s, steps)
        self.create_single_folder(path)
        with mrcfile.new(path + '/groundTruth.mrc', overwrite=True) as mrc:
            mrc.set_data(true[0, ..., 0])
        for k in range(steps + 1):
            with mrcfile.new(path+'/Iteration_'+str(k)+'.mrc', overwrite=True) as mrc:
                mrc.set_data(guess[0, ..., 0])
            guess = self.update_pic(1, step_s, guess)


    def train(self, steps):
        # the training routine
        for k in range(steps):
            if k % 100 == 0:
                self.log_network_training()
            true, estimate = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer,
                          feed_dict={self.gen: estimate, self.true: true})
        self.save(self.global_step)

    def evaluate(self, image):
        # given an image, returns the gradient of the regularization functional
        return self.sess.run(self.pic_grad, feed_dict={self.reconstruction: image})[0]
