import numpy as np
import os
import tensorflow as tf
from ClassFiles.networks import ConvNetClassifier
import ClassFiles.ut as ut
from ClassFiles.ut import fftshift_tf

IMAGE_SIZE = (None, 96, 96, 96, 1)
FOURIER_SIZE = (None, 96, 96, 49, 1)
# Weight on gradient regularization
LMB = 20


class AdversarialRegulariser(object):

    # sets up the network architecture
    def __init__(self, path):

        self.path =path
        self.network = ConvNetClassifier()
        self.sess = tf.InteractiveSession()
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)


        ut.create_single_folder(self.path+'/Data')
        ut.create_single_folder(self.path + '/Logs')

        ### Training the regulariser ###

        # placeholders
        self.fourier_data = tf.placeholder(shape=FOURIER_SIZE, dtype=tf.complex64)
        self.true = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # process the Fourier data
        real_data = tf.expand_dims(tf.spectral.irfft3d(self.fourier_data[...,0]), axis=-1)
        self.gen = fftshift_tf(real_data)

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
        self.loss_was = self.wasserstein_loss + LMB * self.regulariser_was

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        ### The reconstruction network ###
        # placeholders
        self.reconstruction = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.ground_truth = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)

        # the loss functional
        self.was_output = tf.reduce_mean(self.network.net(self.reconstruction))
        self.was_cor = self.was_output - tf.reduce_mean(self.network.net(self.ground_truth))

        # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
        # averaged quantities already. Makes gradients scaling batch size inveriant
        batch_s = tf.cast(tf.shape(self.reconstruction)[0], tf.float32)

        # Optimization for the picture
        self.pic_grad = tf.gradients(self.was_output * batch_s, self.reconstruction)[0]

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
        self.writer = tf.summary.FileWriter(self.path + '/Logs/Network_Optimization/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def evaluate(self, fourierData):
        real_data = ut.irfft(fourierData)
        real_data = ut.unify_form(real_data)
        grad = self.sess.run(self.pic_grad, feed_dict={self.reconstruction: real_data})
        return ut.adjoing_irfft(grad[0,...,0])

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

    # Logging method for minimization. Computes the gradients as 'evaluate', but also writes everything to tensorboard
    # sample id specifies the folder to write to.
    def log_optimization(self, groundTruth, fourierData, id, step):
        groundTruth = ut.unify_form(groundTruth)
        real_data = ut.irfft(fourierData)
        real_data = ut.unify_form(real_data)
        writer = tf.summary.FileWriter(self.path + '/Logs/Picture_Opt/' + id)
        summary, grad = self.sess.run([self.merged_pic, self.pic_grad],
                                feed_dict={self.reconstruction: real_data,
                                           self.ground_truth: groundTruth})
        writer.add_summary(summary, step)
        writer.flush()
        return ut.adjoing_irfft(grad[0,...,0])


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
