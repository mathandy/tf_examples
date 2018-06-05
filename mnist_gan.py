"""
Finished... but doesn't work so well.
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from imageio import imwrite
import os

dropout = tf.layers.dropout
conv2d = tf.layers.conv2d
max_pooling2d = tf.layers.max_pooling2d
dense = tf.layers.dense
flatten = tf.layers.flatten
relu = tf.nn.relu


# Dataset Parameters
num_classes = 10
img_shape = (28, 28, 1)

# Training Parameters (only used if npz file isn't present)
TRAINING_LEARNING_RATE = 0.001
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 10
TRAINING_DROPOUT = 0.25

# Generator Parameters
LEARNING_RATE = 0.001
DROPOUT = 0.25
USE_DROPOUT = True  # try true also!!!!!!!!!!!!!!!!!!
use_loss2 = False
max_steps = int(1e5)
step_per_report = 100
step_per_image_write = 1000
results_dir = '/home/andy/Desktop/mnist_inversion_results'


_network_parameter_names = ['conv1/bias', 'conv1/kernel',
                            'conv2/bias', 'conv2/kernel',
                            'fc1/bias', 'fc1/kernel',
                            'fc2/bias', 'fc2/kernel']


##########################################################


def discriminator(input_tensor, img_shape, num_classes, is_training, droprate=.25):
    # x = tf.placeholder(tf.float32, shape=(-1,) + img_shape)
    x = tf.reshape(input_tensor, shape=(-1,) + img_shape)

    net = conv2d(x, 32, 3, activation=relu, name='conv1')
    net = max_pooling2d(net, 2, 2, name='pool1')

    net = conv2d(net, 64, 3, activation=relu, name='conv2')
    net = max_pooling2d(net, 2, 2, name='pool2')
    net = dropout(net, rate=droprate, training=is_training)

    net = flatten(net)

    net = dense(net, 128, activation=relu, name='fc1')
    net = dropout(net, rate=droprate, training=is_training)

    logits = dense(net, num_classes, name='fc2')
    return logits, x


def discriminator(input_tensor, img_shape, num_classes, is_training, droprate=.25):
    # x = tf.placeholder(tf.float32, shape=(-1,) + img_shape)
    x = tf.reshape(input_tensor, shape=(-1,) + img_shape)

    net = conv2d(x, 32, 3, activation=relu, name='conv1')
    net = max_pooling2d(net, 2, 2, name='pool1')

    net = conv2d(net, 64, 3, activation=relu, name='conv2')
    net = max_pooling2d(net, 2, 2, name='pool2')
    net = dropout(net, rate=droprate, training=is_training)

    net = flatten(net)

    net = dense(net, 128, activation=relu, name='fc1')
    net = dropout(net, rate=droprate, training=is_training)

    logits = dense(net, num_classes, name='fc2')
    return logits, x


def model_fcn(params, labels, use_loss2=True):
    logits, x = reparameterized_network(USE_DROPOUT, params)
    y_hat = tf.nn.softmax(logits)

    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels))

    factor = tf.constant(1e-2)
    loss2 = factor*tf.reduce_mean(x)

    if use_loss2:
        loss = loss1 - loss2
    else:
        loss = loss1

    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss, global_step=tf.train.get_global_step())

    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                              predictions=tf.argmax(y_hat, axis=1))

    return acc, loss, loss1, loss2, train_op, y_hat, x


##########################################################

def save_images(filename, images):
    images -= images.min()
    images /= images.max()
    images *= 255
    images = images.astype('uint8')

    image_bar = np.concatenate([img for img in images])
    imwrite(filename, image_bar)
    return images


def generate_examples(pretrained_params, labels, use_loss2=True):
    with tf.Session() as sess:

        acc, loss, loss1, loss2, train_op, y_hat, x = \
            reparameterized_model_fcn(pretrained_params, labels, use_loss2)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for step in range(max_steps):
            _ = sess.run(fetches=[train_op])

            if not (step % step_per_report):
                acc_, loss1_, loss2_ = sess.run(fetches=[acc, loss1, loss2])
                print("step: %s | loss1 = %s | loss2 = %s |acc = %s" % (step, loss1_, loss2_, acc_))

            if not (step % step_per_image_write):
                save_images(os.path.join(results_dir, 'step_%s.jpg'%step),
                            sess.run(x))

        return sess.run(x)


if __name__ == '__main__':
    try:
        pretrained_parameters = np.load('cnn_mnist_parameters.npz')
    except FileNotFoundError:
        pretrained_parameters = pretrain_parameters()
        np.savez('cnn_mnist_parameters', **pretrained_parameters)

    pretrained_parameters = dict(pretrained_parameters)
    generated_images = generate_examples(pretrained_parameters,
                                         np.identity(num_classes),
                                         use_loss2=use_loss2)
