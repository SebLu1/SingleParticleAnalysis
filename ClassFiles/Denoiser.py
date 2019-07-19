import os
import tensorflow as tf
from ClassFiles.networks import UNet
import ClassFiles.ut as ut
# from ClassFiles.ut import fftshift_tf
from ClassFiles.ut import normalize_tf, sobolev_norm, normalize_np

IMAGE_SIZE = (None, 96, 96, 96, 1)
FOURIER_SIZE = (None, 96, 96, 49, 1)


def data_augmentation_default(gt, adv, noise_lvl):
    return gt, adv


class Denoiser(object):

    def __init__(self, path, data_augmentation=data_augmentation_default, solver='Adam', load=True, cp=None, s=0.0, cutoff=20.0, normalize='l2'):
#        self.noise_lvl = 1.0
        self.path = path
        self.network = UNet()
        self.sess = tf.InteractiveSession()
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.solver = solver
        self.normalize = normalize

        ut.create_single_folder(self.path + '/Data')
        ut.create_single_folder(self.path + '/Logs')

        ### Training the denoiser ###
        self.data = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.true = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

#         Network outputs
        if self.normalize == 'l2':
            self.true_normed, self.data_normed = data_augmentation(normalize_tf(self.true),
                                                         normalize_tf(self.data))
        elif self.normalize == 'NO':
            self.true_normed, self.data_normed = data_augmentation(self.true, self.data)#, self.noise_lvl)
    
            self.denoised = self.network.net(self.data_normed)

        # Loss
        if s == 0.0:
 #           print('good')
 #           raise Exception
            self.loss = 0.5 * tf.reduce_sum((self.true_normed - self.denoised) ** 2)
#            print('bad')
        else:
            self.loss = 0.5 * tf.reduce_sum(sobolev_norm(self.true_normed - self.denoised, s=s, cutoff=cutoff) ** 2)
        
        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.solver == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        if self.solver == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        if self.solver == 'Mom':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        # Logging tools
        summary_list = []
        with tf.name_scope('Network_Optimization'):
            summary_list.append(tf.summary.scalar('Loss', self.loss))
 #           l.append(tf.summary.scalar('Data_Difference', self.wasserstein_loss))
 #           l.append(tf.summary.scalar('Lipschitz_Regulariser', self.regulariser_was))
 #           l.append(tf.summary.scalar('Overall_Net_Loss', self.loss_was))
 #           l.append(tf.summary.scalar('Norm_Input_true', tf.norm(self.true)))
 #           l.append(tf.summary.scalar('Norm_Input_adv', tf.norm(self.gen)))
 #           l.append(tf.summary.scalar('Norm_Gradient', tf.norm(self.gradient)))
            with tf.name_scope('Maximum_Projection'):
                summary_list.append(tf.summary.image('Noisy', tf.reduce_max(self.data_normed, axis=3), max_outputs=1))
                summary_list.append(tf.summary.image('Ground_Truth', tf.reduce_max(self.true_normed, axis=3), max_outputs=1))
                summary_list.append(tf.summary.image('Denoised', tf.reduce_max(tf.abs(self.denoised), axis=3), max_outputs=1))
#                l.append(tf.summary.image('Gradient_GT', tf.reduce_max(tf.abs(gradient_track), axis=3), max_outputs=1))
            slice = int(IMAGE_SIZE[3]/2)
            with tf.name_scope('Slice_Projection'):
                summary_list.append(tf.summary.image('Noisy', self.data_normed[..., slice, :], max_outputs=1))
                summary_list.append(tf.summary.image('Ground_Truth', self.true_normed[..., slice, :], max_outputs=1))
                summary_list.append(tf.summary.image('Denoised', self.denoised[..., slice, :],  max_outputs=1))
 #               l.append(tf.summary.image('Gradient_GT', gradient_track[..., slice, :], max_outputs=1))

            self.merged_network = tf.summary.merge(summary_list)

        # set up the logger
        self.writer_train = tf.summary.FileWriter(self.path + '/Logs/Network_Optimization/TrainingData/')
        self.writer_test = tf.summary.FileWriter(self.path + '/Logs/Network_Optimization/TestData/')

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        if load:
            self.load(cp)

    def evaluate(self, data):
        data_uf = ut.unify_form(data)
        norm = 1.0
        if self.normalize == 'l2':
            norm, data_uf = normalize_np(data_uf, return_norm=True)
        elif self.normalize == 'NO':
            pass
        return norm * self.denoised.eval(feed_dict={self.data_normed: data_uf})[0, ..., 0]

    def train(self, groundTruth, noisy, learning_rate=None): # noise_lvl,
#        self.noise_lvl = noise_lvl
#        print('In train_den:', self.noise_lvl)
        groundTruth = ut.unify_form(groundTruth)
        noisy = ut.unify_form(noisy)
        self.sess.run(self.optimizer,
                      feed_dict={self.true: groundTruth,
                                 self.data: noisy,
                                 self.learning_rate: learning_rate})#,
#                                 self.noise_lvl: noise_lvl})

    # Input as in 'train', but writes results to tensorboard instead
    def test(self, groundTruth, noisy, writer='train'): # noise_levle
#        self.noise_lvl = noise_lvl
        groundTruth = ut.unify_form(groundTruth) #/ (noise_lvl * 500)
        noisy = ut.unify_form(noisy) #/ (noise_lvl * 500)
        merged, step = self.sess.run([self.merged_network, self.global_step],
                                     feed_dict={self.true: groundTruth,
                                                self.data: noisy})
        if writer == 'train':
            self.writer_train.add_summary(merged, global_step=step)

        if writer == 'test':
            self.writer_test.add_summary(merged, global_step=step)            
            
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path + '/Data/model',
                   global_step=self.global_step)
        print('Progress saved')

    def load(self, cp):
        saver = tf.train.Saver()
        if cp == None:
            if os.listdir(self.path + '/Data/'):
                saver.restore(self.sess,
                              tf.train.latest_checkpoint(self.path + '/Data/'))
                print('Save restored')
            else:
                print('No save found')
        else:
            if os.listdir(cp):
                saver.restore(self.sess,
                              tf.train.latest_checkpoint(cp))
                print('Save restored')
            else:
                print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()
