from __future__ import division, print_function, absolute_import
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from scipy.io import loadmat
from andnn import AnDNNClassifier
from andnn.iotools import k21hot, shuffle_together, split_data
from andnn.layers import fc_layer, conv_layer
from andnn.losses import ce_wlogits
from andnn.utils import step_plot, accuracy, num_correct, num_incorrect, batches


def vgg16(X, num_classes=10):
    parameters = []  # storage for trainable parameters

    # pooling arguments
    _ksize = [1, 2, 2, 1]
    _strides = [1, 2, 2, 1]

    # Inputs to be fed in at each step
    # with tf.name_scope('input'):
    #     X = tf.placeholder(dtype,
    #                        shape=np.append(batch_size, im_shape),
    #                        name='X_input')
    #     Y = tf.placeholder(dtype,
    #                        shape=np.append(batch_size, num_classes),
    #                        name='Y_input')

    # # center the input images
    # with tf.name_scope('preprocess_centering'):
    #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
    #                        shape=[1, 1, 1, 3], name='img_mean')
    #     c_images = X - mean
    c_images = X

    # images --> conv1_1 --> conv1_2 --> pool1
    conv1_1, weights1, biases1 = conv_layer(c_images, 3, 3, 64, 'conv1_1')
    conv1_2, weights2, biases2 = conv_layer(conv1_1, 3, 64, 64, 'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')
    # parameters += [weights1, biases1, weights2, biases2]

    # pool1 --> conv2_1 --> conv2_2 --> pool2
    conv2_1, weights1, biases1 = conv_layer(pool1, 3, 64, 128, 'conv2_1')
    conv2_2, weights2, biases2 = conv_layer(conv2_1, 3, 128, 128, 'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')
    # parameters += [weights1, biases1, weights2, biases2]

    # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
    conv3_1, weights1, biases1 = conv_layer(pool2, 3, 128, 256, 'conv3_1')
    conv3_2, weights2, biases2 = conv_layer(conv3_1, 3, 256, 256, 'conv3_2')
    conv3_3, weights3, biases3 = conv_layer(conv3_2, 3, 256, 256, 'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')
    # parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
    conv4_1, weights1, biases1 = conv_layer(pool3, 3, 256, 512, 'conv4_1')
    conv4_2, weights2, biases2 = conv_layer(conv4_1, 3, 512, 512, 'conv4_2')
    conv4_3, weights3, biases3 = conv_layer(conv4_2, 3, 512, 512, 'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')
    # parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
    conv5_1, weights1, biases1 = conv_layer(pool4, 3, 512, 512, 'conv5_1')
    conv5_2, weights2, biases2 = conv_layer(conv5_1, 3, 512, 512, 'conv5_2')
    conv5_3, weights3, biases3 = conv_layer(conv5_2, 3, 512, 512, 'conv5_3')
    pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')
    # parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool5 --> flatten --> fc1 --> fc2 --> fc3
    pool5_out_dim = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, pool5_out_dim])
    # fc1, weights1, biases1 = fc_layer(pool5_flat, pool5_out_dim, 4096, name='fc1')
    # fc2, weights2, biases2 = fc_layer(fc1, 4096, 4096, name='fc2')
    # fc3pre, weights3, biases3 = fc_layer(fc2, 4096, num_classes, None, 'fc3pre')
    # fc3, weights3, biases3 = fc_layer(fc2, 4096, num_classes, tf.nn.softmax, 'fc3')
    fc1 = fully_connected(pool5_flat, 4096, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = fully_connected(fc1, 4096, activation_fn=tf.nn.relu, scope='fc2')
    fc3pre = fully_connected(fc2, num_classes, activation_fn=None, scope='fc3pre')

    # parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
    # activations = {
    #     'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
    #     'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
    #     'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3,
    #     'pool3': pool3,
    #     'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3,
    #     'pool4': pool4,
    #     'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3,
    #     'pool5': pool5,
    #     'fc1': fc1, 'fc2': fc2, 'fc3': fc3
    # }
    return fc3pre


if __name__ == '__main__':
    usps_data = loadmat('usps/USPS.mat')
    X, Y = usps_data['fea'], usps_data['gnd']
    X = X.reshape(-1, 16, 16)
    Y = k21hot(Y)

    X = np.stack((X, X, X), axis=3)  # pretend USPS is colored

    Xtrain, Ytrain, Xvalid, Yvalid, _, _ = \
        split_data(X, Y, validpart=.2, testpart=0)
    # (X, Y), permutation = shuffle_together((X, Y))

    classifier = AnDNNClassifier(vgg16,
                                 final_activation=tf.nn.softmax,
                                 example_shape=X.shape[1:],
                                 label_shape=Y.shape[1:],
                                 debug=False)
    classifier.fit(X, Y, batch_size=500, epochs=20,
                   loss=ce_wlogits,
                   loss_kwargs={},
                   optimizer=tf.train.AdamOptimizer,
                   optimizer_kwargs={'learning_rate': 1e-5},
                   steps_per_report=500,
                   X_valid=Xvalid, Y_valid=Yvalid)

    # if debug:
    #     print([v.name for v in tf.trainable_variables()])
