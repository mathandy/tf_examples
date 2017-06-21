from __future__ import division, print_function, absolute_import
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from scipy.io import loadmat
from andnn import AnDNNClassifier, ce_wlogits
from andnn.iotools import k21hot, shuffle_together, split_data
from andnn.layers import fc_layer
from andnn.losses import ce_wlogits
from andnn.utils import step_plot, accuracy, num_correct, num_incorrect, batches


# def multilayer_network(x):
#     fc1, _, _ = fc_layer(x, 256, 256, act=tf.nn.relu, name='fc1')
#     fc2, _, _ = fc_layer(fc1, 256, 256, act=tf.nn.relu, name='fc2')
#     fc3pre, _, _ = fc_layer(fc2, 256, 10, act=None, name='fc3pre')
#     return fc3pre


def multilayer_network(x):
    fc1 = fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc3pre = fully_connected(fc2, 10, activation_fn=None, scope='fc3pre')
    return fc3pre


def multilayer_network_with_dropout(x):
    keep_prob = 0.5
    fc1 = fully_connected(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc1d = tf.nn.dropout(fc1, keep_prob, scope='fc1-dropout')
    fc2 = fully_connected(fc1d, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2d = tf.nn.dropout(fc2, keep_prob, scope='fc2-dropout')
    fc3pre = fully_connected(fc2d, 10, activation_fn=None, scope='fc3pre')
    return fc3pre

    
if __name__ == '__main__':
    usps_data = loadmat('usps/USPS.mat')
    X, Y = usps_data['fea'], usps_data['gnd']
    # X = X.reshape(-1, 16, 16)
    Y = k21hot(Y)
    # Y = Y.sum(axis=1).reshape(-1, 1)

    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = \
        split_data(X, Y, validpart=.2, testpart=.2)
    # (X, Y), permutation = shuffle_together((X, Y))

    classifier = AnDNNClassifier(multilayer_network,
                            final_activation=tf.nn.softmax,
                            example_shape=X.shape[1:],
                            label_shape=Y.shape[1:],
                            debug=False)
    classifier.fit(X, Y, batch_size=500, epochs=20,
                   loss=ce_wlogits,
                   loss_kwargs=None,
                   optimizer=tf.train.AdamOptimizer,
                   optimizer_kwargs={'learning_rate': 1e-7},
                   steps_per_report=len(Xtrain),  # report every epoch
                   X_valid=Xvalid, Y_valid=Yvalid)


    # if debug:
    #     print([v.name for v in tf.trainable_variables()])
