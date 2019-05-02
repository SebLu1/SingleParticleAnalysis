import tensorflow as tf
from tensorflow.python.keras.layers import UpSampling3D
from abc import ABC, abstractmethod

def lrelu(x):
    # leaky rely
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

# The format the networks for reconstruction are written in. Size in format (width, height) gives the shape
# of an input image. colors specifies the amount of output channels for image to image architectures.
class network(ABC):

    # Method defining the neural network architecture, returns computation result. Use reuse=tf.AUTO_REUSE.
    @abstractmethod
    def net(self, input):
        pass

### basic network architectures ###
# A couple of small network architectures for computationally light comparison experiments.
# No dropout, no batch_norm, no skip-connections


class ConvNetClassifier(network):
    # classical classifier with convolutional layers with strided convolutions and two dense layers at the end

    def net(self, inp):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv3d(inputs=inp, filters=16, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv1')
        conv2 = tf.layers.conv3d(inputs=conv1, filters=16, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv2')
        conv3 = tf.layers.conv3d(inputs=conv2, filters=16, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv3', strides=2)
        # image size is now size/2
        conv4 = tf.layers.conv3d(inputs=conv3, filters=32, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv4', strides=2)
        # image size is now size/4
        conv5 = tf.layers.conv3d(inputs=conv4, filters=32, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv5', strides=2)
        # image size is now size/8
        conv6 = tf.layers.conv3d(inputs=conv5, filters=64, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv6', strides=2)
        # image size is now size/16
        conv7 = tf.layers.conv3d(inputs=conv6, filters=64, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv7', strides=2)

        # reshape for classification
        reshaped = tf.layers.flatten(conv7)

        # dense layer for classification
        dense = tf.layers.dense(inputs=reshaped, units=256, activation=lrelu, reuse=tf.AUTO_REUSE, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=tf.AUTO_REUSE, name='dense2')

        # Output network results
        return output


def apply_conv(x, filters=32, kernel_size=3, he_init=True, factor=2.0, cst_ini=False):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=factor)
    else:
        initializer = tf.contrib.layers.xavier_initializer()

    if cst_ini:
        initializer = tf.constant_initializer(value=1e-2)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

    return tf.layers.conv3d(x,
                            filters=filters, kernel_size=kernel_size,
                            padding='SAME',
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer)


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.leaky_relu(x)


def meanpool(x):
    with tf.name_scope('meanpool'):
        return tf.layers.average_pooling3d(
            x,
            2,
            2,
            padding='same',
            data_format='channels_last',
        )


def upsample(x):
    l = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
    with tf.name_scope('upsample'):
        return l(x)


def resblock(x, filters, he_init=False, factor=2.0):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
#        update = apply_conv(activation(x), filters=filters, he_init=he_init, factor=factor)
#        update = apply_conv(activation(update), filters=filters, he_init=he_init, factor=factor)
        update = apply_conv(activation(x), filters=filters)
        update = apply_conv(activation(update), filters=filters)
        
#        skip = apply_conv(x, filters=filters, kernel_size=1, he_init=he_init, factor=factor, cst_ini=True)
        skip = apply_conv(x, filters=filters, kernel_size=1, he_init=he_init, factor=factor)

        return skip + update


class ResNetClassifierL2(network):
    def net(self, x_in):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.name_scope('pre_process'):
                x0 = apply_conv(x_in, filters=16, kernel_size=3)

            with tf.name_scope('x1'):
                x1 = resblock(x0, 16)  # 96

            with tf.name_scope('x2'):
                x2 = resblock(meanpool(x1), filters=32)  # 48

            with tf.name_scope('x3'):
                x3 = resblock(meanpool(x2), filters=64)  # 24

            with tf.name_scope('x4'):
                x4 = resblock(meanpool(x3), filters=128)  # 12
                
            with tf.name_scope('x5'):
                x5 = resblock(meanpool(x4), filters=128)  # 6
            
            with tf.name_scope('post_process'):
                base_case = tf.sqrt(tf.reduce_sum(x_in ** 2, axis=[1, 2, 3, 4]))
                
            with tf.name_scope('flat'):
                flat = tf.contrib.layers.flatten(x5)
                flat = tf.layers.dense(flat, 128, activation=activation)
                flat = tf.layers.dense(flat, 1)
                
            with tf.name_scope('return'):
                return base_case + flat


class ResNetClassifier(network):
    def net(self, x_in):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.name_scope('pre_process'):
                x0 = apply_conv(x_in, filters=16, kernel_size=3)

            with tf.name_scope('x1'):
                x1 = resblock(x0, 16)  # 96

            with tf.name_scope('x2'):
                x2 = resblock(meanpool(x1), filters=32)  # 48

            with tf.name_scope('x3'):
                x3 = resblock(meanpool(x2), filters=64)  # 24

            with tf.name_scope('x4'):
                x4 = resblock(meanpool(x3), filters=128)  # 12

            with tf.name_scope('x5'):
                x5 = resblock(meanpool(x4), filters=128)  # 6

            with tf.name_scope('flat'):
                flat = tf.contrib.layers.flatten(x5)
                flat = tf.layers.dense(flat, 128, activation=activation)
                flat = tf.layers.dense(flat, 1)

            with tf.name_scope('return'):
                return flat


class UNet(network):

    def net(self, x_in):
        with tf.variable_scope('UNet', reuse=tf.AUTO_REUSE):
            with tf.name_scope('pre_process'):
                x0 = apply_conv(x_in, filters=16, kernel_size=3, he_init=True, factor=1e-2)

            with tf.name_scope('x1'):
                x1 = resblock(x0, 16, he_init=True, factor=1e-2)  # 96

            with tf.name_scope('x2'):
                x2 = resblock(meanpool(x1), filters=32, he_init=True, factor=1e-2)  # 48

            with tf.name_scope('x3'):
                x3 = resblock(meanpool(x2), filters=64, he_init=True, factor=1e-2)  # 24

            with tf.name_scope('x4'):
                x4 = resblock(meanpool(x3), filters=64, he_init=True, factor=1e-2)  # 12

            with tf.name_scope('x5'):
                x5 = resblock(meanpool(x4), filters=128, he_init=True, factor=1e-2)  # 6

            with tf.name_scope('x6'):
                x6 = resblock(upsample(x5), filters=64, he_init=True, factor=1e-2)  # 12

            with tf.name_scope('x7'):
                x7 = resblock(upsample(tf.concat([x4, x6], -1)), filters=64, he_init=True, factor=1e-2)  # 24

            with tf.name_scope('x8'):
                x8 = resblock(upsample(tf.concat([x3, x7], -1)), filters=32, he_init=True, factor=1e-2)  # 48

            with tf.name_scope('x9'):
                x9 = resblock(upsample(tf.concat([x2, x8], -1)), filters=16, he_init=True, factor=1e-2)  # 96

            with tf.name_scope('x10'):
                x10 = resblock(tf.concat([x1, x9], -1), filters=16, he_init=True, factor=1e-2)

            with tf.name_scope('post_process'):
                return apply_conv(x10, filters=1, kernel_size=3, he_init=True, factor=1e-2)





