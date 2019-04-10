from ClassFiles import tensorflow_rotations
import tensorflow as tf
import numpy as np


class Registrator(object):
    IMAGE_SIZE = (None, 96, 96, 96, 1)
    NOISE = 0.01
    IMAX = 200
    LEARNING_RATE = 1e-2

    def __init__(self, batch_size=1):
        with tf.variable_scope('Registrator'):
            self.batch_size=batch_size
            self.sess = tf.InteractiveSession()

            self.image_feed = tf.placeholder(shape=self.IMAGE_SIZE, dtype=tf.float32)
            self.reference_feed = tf.placeholder(shape=self.IMAGE_SIZE, dtype=tf.float32)
            self.noise = tf.placeholder(shape=(), dtype=tf.float32)

            basis_exp = tf.get_variable(name='Rotations', shape=[batch_size, 3, 3], 
                                        initializer = tf.initializers.zeros)
            skew_exp = basis_exp - tf.transpose(basis_exp, perm=[0,2,1])
            self.rotation = tf.linalg.expm(skew_exp)
            self.translation = tf.get_variable(name='Translations', shape=[batch_size, 3, 1], 
                                               initializer = tf.initializers.zeros)
            self.theta = tf.concat([self.rotation, self.translation], axis=-1)

            self.rot_image = tensorflow_rotations.rot3d(self.image_feed, self.theta)

            diff = self.rot_image - self.reference_feed
            self.data_fit = tf.reduce_sum(tf.square(diff))
            for i in range(1, 3):
                self.data_fit += tf.reduce_sum(tf.square(tf.layers.average_pooling3d(diff, 2 ** i, 2 ** i)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.data_fit)

            with tf.control_dependencies([self.optimizer]):
                self.assign = tf.group(tf.assign_add(basis_exp, np.pi * self.noise * tf.random_normal(shape=(batch_size, 3, 3))),
                                tf.assign_add(self.translation, self.noise * tf.random_normal(shape=(batch_size, 3, 1))))
        
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Registrator')
        self.initializer = tf.initializers.variables(var_list=var)
        self.initializer.run()

    def register(self, image, reference, printing=True):
        # Can only feed input that fits the batch size the graph has been generated with
        assert image.shape[0]==self.batch_size
        self.initializer.run()

        for k in range(self.IMAX):
            _, _, loss = self.sess.run([self.optimizer, self.assign, self.data_fit],
                                                   feed_dict={self.image_feed: image,
                                                              self.reference_feed: reference,
                                                              self.noise: self.NOISE * (k < self.IMAX // 2)})
            if k % 10 == 0:
                print(loss)

        result, theta, rot, trans, loss = self.sess.run([self.rot_image, self.theta, self.rotation,
                                                         self.translation, self.data_fit],
                                                           feed_dict={self.image_feed: image,
                                                                      self.reference_feed: reference})

        if printing:
            print('Final Registration Loss: ' + str(loss))
            print('Rotation: ')
            print(rot)
            print('Translation: ')
            print(trans)

        return result


class LocalRegistrator(Registrator):

    NOISE = 0.001
    IMAX = 50
    LEARNING_RATE = 1e-2



