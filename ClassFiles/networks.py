import tensorflow as tf
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

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv3d(inputs=input, filters=16, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv1')
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv2')
        conv3 = tf.layers.conv3d(inputs=conv2, filters=32, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv3', strides=2)
        # image size is now size/2
        conv4 = tf.layers.conv3d(inputs=conv3, filters=64, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv4', strides=2)
        # image size is now size/4
        conv5 = tf.layers.conv3d(inputs=conv4, filters=64, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv5', strides=2)
        # image size is now size/8
        conv6 = tf.layers.conv3d(inputs=conv5, filters=128, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv6', strides=2)
        # image size is now size/16
        conv7 = tf.layers.conv3d(inputs=conv6, filters=128, kernel_size=[3, 3, 3], padding="same",
                                 activation=lrelu, reuse=tf.AUTO_REUSE, name='conv7', strides=2)

        # reshape for classification
        reshaped = tf.layers.flatten(conv7)

        # dense layer for classification
        dense = tf.layers.dense(inputs=reshaped, units=256, activation=lrelu, reuse=tf.AUTO_REUSE, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=tf.AUTO_REUSE, name='dense2')

        # Output network results
        return output



