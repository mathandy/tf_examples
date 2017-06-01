from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from andnn.layers import fc_layer, conv_layer


def vgg16(X_batch_shape, Y_batch_shape):

    """
    
    Parameters
    ----------
    X_batch_shape
    Y_batch_shape
    for_training
    return_logits

    Returns
    -------

    """
    parameters = []  # storage for trainable parameters

    # pooling arguments
    _ksize = [1, 2, 2, 1]
    _strides = [1, 2, 2, 1]

    # derived definitions
    K = Y.shape[1]

    # Inputs to be fed in at each step
    with tf.name_scope('input'):
        X = tf.placeholder(dtype,
                           shape=np.append(batch_size, im_shape),
                           name='X_input')
        Y = tf.placeholder(dtype,
                           shape=np.append(batch_size, num_classes),
                           name='Y_input')

    # # center the input images
    # with tf.name_scope('preprocess_centering'):
    #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
    #                        shape=[1, 1, 1, 3], name='img_mean')
    #     c_images = X - mean

    # images --> conv1_1 --> conv1_2 --> pool1
    conv1_1, weights1, biases1 = conv_layer(c_images, 3, 3, 64, 'conv1_1')
    conv1_2, weights2, biases2 = conv_layer(conv1_1, 3, 64, 64, 'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')
    parameters += [weights1, biases1, weights2, biases2]

    # pool1 --> conv2_1 --> conv2_2 --> pool2
    conv2_1, weights1, biases1 = conv_layer(pool1, 3, 64, 128, 'conv2_1')
    conv2_2, weights2, biases2 = conv_layer(conv2_1, 3, 128, 128, 'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')
    parameters += [weights1, biases1, weights2, biases2]

    # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
    conv3_1, weights1, biases1 = conv_layer(pool2, 3, 128, 256, 'conv3_1')
    conv3_2, weights2, biases2 = conv_layer(conv3_1, 3, 256, 256, 'conv3_2')
    conv3_3, weights3, biases3 = conv_layer(conv3_2, 3, 256, 256, 'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')
    parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
    conv4_1, weights1, biases1 = conv_layer(pool3, 3, 256, 512, 'conv4_1')
    conv4_2, weights2, biases2 = conv_layer(conv4_1, 3, 512, 512, 'conv4_2')
    conv4_3, weights3, biases3 = conv_layer(conv4_2, 3, 512, 512, 'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')
    parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
    conv5_1, weights1, biases1 = conv_layer(pool4, 3, 512, 512, 'conv5_1')
    conv5_2, weights2, biases2 = conv_layer(conv5_1, 3, 512, 512, 'conv5_2')
    conv5_3, weights3, biases3 = conv_layer(conv5_2, 3, 512, 512, 'conv5_3')
    pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')
    parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

    # pool5 --> flatten --> fc1 --> fc2 --> fc3
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1, weights1, biases1 = fc_layer(pool5_flat, shape, 4096, name='fc1')
    fc2, weights2, biases2 = fc_layer(fc1, 4096, 4096, name='fc2')
    fc3pre, weights3, biases3 = fc_layer(fc2, 4096, K, None, 'fc3')
    fc3, weights3, biases3 = fc_layer(fc2, 4096, K, tf.nn.softmax, 'fc3')

    parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
    activations = {
        'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
        'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
        'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3,
        'pool3': pool3,
        'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3,
        'pool4': pool4,
        'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3,
        'pool5': pool5,
        'fc1': fc1, 'fc2': fc2, 'fc3': fc3
    }

    # with tf.name_scope('loss'):
    #     if loss_expects_logits:
    #         loss = loss_fcn(fc3pre, Y)
    #     else:
    #         loss = loss_fcn(fc3, Y)
    #
    # with tf.name_scope('training'):
    #       # ?maybe need to reduce again
    #     tf.summary.scalar("loss", loss)
    #     update_W_and_b = optimizer.minimize(loss)
    #
    # full_summary = tf.summary.merge_all()
    #
    # model_dict = {'activations': activations, 'parameters': parameters,
    #               'loss': loss, 'X': X, 'Y': Y, 'step_forward': update_W_and_b,
    #               'update_summary': full_summary}

    return activations, parameters


# def vgg16_old(im_shape, num_classes, batch_size, for_training, optimizer,
#           loss_fcn, dtype=tf.float32, loss_expects_logits=True):
#     """
#
#     Parameters
#     ----------
#     im_shape
#     num_classes
#     batch_size
#     for_training
#     optimizer
#     loss_fcn
#         fcn of form f(x,y)
#     dtype
#     loss_expects_logits
#         set true if loss_fcn expects logits for x
#
#     Returns
#     -------
#
#     """
#     parameters = []  # storage for trainable parameters
#
#     # pooling arguments
#     _ksize = [1, 2, 2, 1]
#     _strides = [1, 2, 2, 1]
#
#     # Inputs to be fed in at each step
#     with tf.name_scope('input'):
#         X = tf.placeholder(dtype,
#                            shape=np.append(batch_size, im_shape),
#                            name='X_input')
#         Y = tf.placeholder(dtype,
#                            shape=np.append(batch_size, num_classes),
#                            name='Y_input')
#
#     # center the input images
#     with tf.name_scope('preprocess_centering'):
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
#                            shape=[1, 1, 1, 3], name='img_mean')
#         c_images = X - mean
#
#     # images --> conv1_1 --> conv1_2 --> pool1
#     conv1_1, weights1, biases1 = conv_layer(c_images, 3, 3, 64, 'conv1_1')
#     conv1_2, weights2, biases2 = conv_layer(conv1_1, 3, 64, 64, 'conv1_2')
#     pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')
#     parameters += [weights1, biases1, weights2, biases2]
#
#     # pool1 --> conv2_1 --> conv2_2 --> pool2
#     conv2_1, weights1, biases1 = conv_layer(pool1, 3, 64, 128, 'conv2_1')
#     conv2_2, weights2, biases2 = conv_layer(conv2_1, 3, 128, 128, 'conv2_2')
#     pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')
#     parameters += [weights1, biases1, weights2, biases2]
#
#     # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
#     conv3_1, weights1, biases1 = conv_layer(pool2, 3, 128, 256, 'conv3_1')
#     conv3_2, weights2, biases2 = conv_layer(conv3_1, 3, 256, 256, 'conv3_2')
#     conv3_3, weights3, biases3 = conv_layer(conv3_2, 3, 256, 256, 'conv3_3')
#     pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')
#     parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
#
#     # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
#     conv4_1, weights1, biases1 = conv_layer(pool3, 3, 256, 512, 'conv4_1')
#     conv4_2, weights2, biases2 = conv_layer(conv4_1, 3, 512, 512, 'conv4_2')
#     conv4_3, weights3, biases3 = conv_layer(conv4_2, 3, 512, 512, 'conv4_3')
#     pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')
#     parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
#
#     # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
#     conv5_1, weights1, biases1 = conv_layer(pool4, 3, 512, 512, 'conv5_1')
#     conv5_2, weights2, biases2 = conv_layer(conv5_1, 3, 512, 512, 'conv5_2')
#     conv5_3, weights3, biases3 = conv_layer(conv5_2, 3, 512, 512, 'conv5_3')
#     pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')
#     parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
#
#     # pool5 --> flatten --> fc1 --> fc2 --> fc3
#     shape = int(np.prod(pool5.get_shape()[1:]))
#     pool5_flat = tf.reshape(pool5, [-1, shape])
#     fc1, weights1, biases1 = fc_layer(pool5_flat, shape, 4096, name='fc1')
#     fc2, weights2, biases2 = fc_layer(fc1, 4096, 4096, name='fc2')
#     fc3pre, weights3, biases3 = fc_layer(fc2, 4096, num_classes, None, 'fc3')
#     fc3, weights3, biases3 = fc_layer(fc2, 4096, num_classes, tf.nn.softmax, 'fc3')
#
#     parameters += [weights1, biases1, weights2, biases2, weights3, biases3]
#     activations = {
#         'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
#         'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
#         'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3,
#         'pool3': pool3,
#         'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3,
#         'pool4': pool4,
#         'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3,
#         'pool5': pool5,
#         'fc1': fc1, 'fc2': fc2, 'fc3': fc3
#     }
#
#     with tf.name_scope('loss'):
#         if loss_expects_logits:
#             loss = loss_fcn(fc3pre, Y)
#         else:
#             loss = loss_fcn(fc3, Y)
#
#     # optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
#     # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
#
#     with tf.name_scope('training'):
#         loss = tf.reduce_mean(loss)  #?maybe need to reduce again
#         tf.summary.scalar("loss", loss)
#         update_W_and_b = optimizer.minimize(loss)
#
#     full_summary = tf.summary.merge_all()
#
#     model_dict = {'activations': activations, 'parameters': parameters,
#                   'loss': loss, 'X': X, 'Y': Y, 'step_forward': update_W_and_b,
#                   'update_summary': full_summary}
#
#     return model_dict

def loss_fcn(logits, Y):
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                       logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)  # batch sum
    # tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


if __name__ == '__main__':
    from scipy.io import loadmat
    from andnn.iotools import split_data, k21hot

    usps_data = loadmat('usps/USPS.mat')
    X, Y = usps_data['fea'], usps_data['gnd']
    X = X.reshape(-1, 16, 16)
    Y = k21hot(Y)

    X = np.stack((X, X, X), axis=3)  # pretend USPS is colored

    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = \
        split_data(X, Y, validpart=.2, testpart=.2)

    im_shape = Xtrain.shape[1:]
    num_classes = Ytrain.shape[1]
    for_training = True
    batch_size = 32

    optimizer = tf.train.AdamOptimizer(0.001)



    vgg_dict = vgg16(im_shape, num_classes, batch_size, for_training=True,
                     optimizer=optimizer, loss_fcn=loss_fcn, dtype=tf.float32,
                     loss_expects_logits=True)

    from andnn.andnn import AnDNN
    dnn = AnDNN(vgg_dict, weights_file=None, session=None,
                tensorboard_dir='/tmp/tflogs')
    dnn.fit(Xtrain, Ytrain, batch_size, valid=0.0, run_id='unnamed',
            epochs=10, steps_per_save=500, steps_per_report=1, step_fcn=None)
