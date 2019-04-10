import os
import tensorflow as tf
from ClassFiles.networks import UNet
import ClassFiles.ut as ut
from ClassFiles.ut import fftshift_tf
from ClassFiles.ut import normalize_tf

IMAGE_SIZE = (None, 96, 96, 96, 1)
FOURIER_SIZE = (None, 96, 96, 49, 1)

def data_augmentation_default(gt, adv):
    return gt, adv

class Denoiser(object):

    def __init__(self, path, data_augmentation=data_augmentation_default):

        self.path = path
        self.network = UNet()
        self.sess = tf.InteractiveSession()
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        ut.create_single_folder(self.path + '/Data')
        ut.create_single_folder(self.path + '/Logs')

        ### Training the denoiser ###
        self.data = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.true = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # Network outputs
        true_normed, data_normed = data_augmentation(normalize_tf(self.true), normalize_tf(self.data))    
        self.denoised = self.network.net(data)

        # Loss
        self.loss = 0.5 * tf.reduce_sum((self.true - self.denoised) ** 2)

        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        # Logging tools
        l = []
        with tf.name_scope('Network_Optimization'):
            l.append(tf.summary.scalar('Data_Difference', self.wasserstein_loss))
            l.append(tf.summary.scalar('Lipschitz_Regulariser', self.regulariser_was))
            l.append(tf.summary.scalar('Overall_Net_Loss', self.loss_was))
            l.append(tf.summary.scalar('Norm_Input_true', tf.norm(self.true)))
            l.append(tf.summary.scalar('Norm_Input_adv', tf.norm(self.gen)))
            l.append(tf.summary.scalar('Norm_Gradient', tf.norm(self.gradient)))
            with tf.name_scope('Maximum_Projection'):
                l.append(tf.summary.image('Adversarial', tf.reduce_max(gen_normed, axis=3), max_outputs=1))
                l.append(tf.summary.image('GroundTruth', tf.reduce_max(true_normed, axis=3), max_outputs=1))
                l.append(tf.summary.image('Gradient_Adv', tf.reduce_max(tf.abs(self.gradient), axis=3), max_outputs=1))
                l.append(tf.summary.image('Gradient_GT', tf.reduce_max(tf.abs(gradient_track), axis=3), max_outputs=1))
            slice = int(IMAGE_SIZE[3]/2)
            with tf.name_scope('Slice_Projection'):
                l.append(tf.summary.image('Adversarial', gen_normed[..., slice, :], max_outputs=1))
                l.append(tf.summary.image('GroundTruth', true_normed[..., slice, :], max_outputs=1))
                l.append(tf.summary.image('Gradient_Adv', self.gradient[..., slice, :],  max_outputs=1))
                l.append(tf.summary.image('Gradient_GT', gradient_track[..., slice, :], max_outputs=1))

            self.merged_network = tf.summary.merge(l)

        # set up the logger
        self.writer = tf.summary.FileWriter(self.path + '/Logs/Network_Optimization/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

#    def evaluate(self, fourierData):
#        fourierData = ut.unify_form(fourierData)
#        grad = self.sess.run(self.gradient, feed_dict={self.fourier_data: fourierData})
#        return ut.adjoint_irfft(grad[0,...,0])
    
#    def evaluate_real(self, real_data):
#        real_data = ut.unify_form(real_data)
#        grad = self.sess.run(self.gradient, feed_dict={self.gen: real_data})
#        return (grad[0,...,0])    

    # trains the network with the groundTruths and adversarial exemples given. If Flag fourier_data is false,
    # the adversarial exemples are expected to be in real space
    def train(self, groundTruth, adversarial, learning_rate, fourier_data =True):
        groundTruth = ut.unify_form(groundTruth)
        adversarial = ut.unify_form(adversarial)
        if fourier_data:
            self.sess.run(self.optimizer, feed_dict={self.true: groundTruth, self.fourier_data: adversarial,
                                                     self.learning_rate: learning_rate})
        else:
            self.sess.run(self.optimizer, feed_dict={self.true: groundTruth, self.gen: adversarial,
                                                     self.learning_rate: learning_rate})

    # Input as in 'train', but writes results to tensorboard instead
    def test(self, groundTruth, adversarial, fourier_data =True):
        groundTruth = ut.unify_form(groundTruth)
        adversarial = ut.unify_form(adversarial)
        if fourier_data:
            merged, step = self.sess.run([self.merged_network, self.global_step],
                                         feed_dict={self.true: groundTruth, self.fourier_data: adversarial})
        else:
            merged, step = self.sess.run([self.merged_network, self.global_step], 
                                         feed_dict={self.true: groundTruth, self.gen: adversarial})
        self.writer.add_summary(merged, global_step=step)


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'/Data/model', global_step=self.global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'/Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'/Data/'))
            print('Save restored')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()
