from ClassFiles import tensorflow_rotations
import tensorflow as tf
import numpy as np
from ClassFiles.ut import l2


class Registrator(object):

    def __init__(self, IMAGE_SIZE):
        self.sess = tf.InteractiveSession()

        self.image_size = IMAGE_SIZE
        self.image_feed = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)
        self.reference_feed = tf.placeholder(shape=IMAGE_SIZE, dtype=tf.float32)

        basis_exp = tf.Variable(np.zeros(shape=(3, 3)))
        skew_exp = basis_exp - tf.transpose(basis_exp)
        self.rotation = tf.linalg.expm(skew_exp)
        self.translation = tf.Variable(np.zeros(shape=(3, 1)))
        self.theta = tf.concat([self.rotation, self.translation], axis=-1)

        self.rot_image = tensorflow_rotations.rot3d(self.image_feed, self.theta)
        self.loss = tf.square(self.rot_image - self.reference_feed)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)
        tf.global_variables_initializer().run()

    def register(self, image, reference, printing=True):
        tf.global_variables_initializer().run()

        i = 0
        end = 10000
        prev_theta = np.zeros
        while i <= 10000:
            _, l, theta = self.sess.run([self.optimizer, self.loss, self.theta],
                                 feed_dict={self.image_feed: image, self.reference_feed: reference})
            if i == 0 and printing:
                print('Initial Registration Loss: ' + str(l))
            if l2(theta-prev_theta) < 1e-6 and i > 30:
                i = end

        result, l, rot, trans = self.sess.run([self.rot_image, self.loss, self.rotation, self.translation],
                                    feed_dict={self.image_feed: image, self.reference_feed: reference})

        if printing:
            print('Final Registration Loss: ' + str(l))
            print('Rotation: ')
            print(rot)
            print('Translation: ')
            print(trans)

        return result


