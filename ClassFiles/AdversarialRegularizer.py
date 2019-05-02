import os
import tensorflow as tf
from ClassFiles.networks import ResNetClassifier
import ClassFiles.ut as ut
from ClassFiles.ut import fftshift_tf, normalize_tf, l2_tf, sobolev_norm


IMAGE_SIZE = (None, 96, 96, 96, 1)
FOURIER_SIZE = (None, 96, 96, 49, 1)
# Weight on gradient regularization

def data_augmentation_default(gt, adv):
    return gt, adv

class AdversarialRegulariser(object):
    # sets up the network architecture
    def __init__(self, path, data_augmentation=data_augmentation_default, s=0.0, gamma=1.0, lmb=10.0):
        
        if s == 0.0:
            norm = l2_tf
        else:
            norm = sobolev_norm

        self.path =path
        self.network = ResNetClassifier()
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
        true_normed, gen_normed = data_augmentation(normalize_tf(self.true), normalize_tf(self.gen))
        
        self.gen_was = self.network.net(gen_normed)
        self.data_was = self.network.net(true_normed)

        # Wasserstein loss
        self.wasserstein_loss = tf.reduce_mean(self.data_was - self.gen_was)
                                 
        # Gradient for reconstruction
        self.gradient = tf.gradients(tf.reduce_sum(self.gen_was), gen_normed)[0]

        # Gradient for trakcing
        gradient_track = tf.gradients(tf.reduce_sum(self.data_was), true_normed)[0]

        # intermediate point
        random_int = tf.random_uniform([tf.shape(self.true)[0], 1, 1, 1, 1], 0.0, 1.0)
        self.inter = tf.multiply(gen_normed, random_int) + tf.multiply(true_normed, 1 - random_int)
        self.inter_was = tf.reduce_sum(self.network.net(self.inter))

        # calculate derivative at intermediate point
        self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]

        # take the L2 norm of that derivative
        self.norm_gradient = norm(self.gradient_was)
        self.regulariser_was = tf.reduce_mean(tf.square(self.norm_gradient/gamma - 1))

        # Overall Net Training loss
        self.loss_was = self.wasserstein_loss + lmb * self.regulariser_was

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_was,
                                                                                global_step=self.global_step)

        # logging tools
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

#         with tf.name_scope('Picture_Optimization'):
#             wasser_loss = tf.summary.scalar('Wasserstein_Loss', self.was_cor)
#             recon = tf.summary.image('Reconstruction', self.cut_reco[..., sliceN, :], max_outputs=1)
#             ground_truth = tf.summary.image('Ground_truth', self.ground_truth[..., sliceN, :], max_outputs=1)
#             quality_assesment = tf.summary.scalar('L2_to_ground_truth', self.quality)
#             self.merged_pic = tf.summary.merge([wasser_loss, quality_assesment, recon, ground_truth])

        # set up the logger
        self.writer = tf.summary.FileWriter(self.path + '/Logs/Network_Optimization/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def evaluate(self, fourierData):
        fourierData = ut.unify_form(fourierData)
        grad = self.sess.run(self.gradient, feed_dict={self.fourier_data: fourierData})
        return ut.adjoint_irfft(grad[0,...,0])
    
    def evaluate_real(self, real_data):
        real_data = ut.unify_form(real_data)
        grad = self.sess.run(self.gradient, feed_dict={self.gen: real_data})
        return (grad[0,...,0])    

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
