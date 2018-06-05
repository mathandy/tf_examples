"""
An experiment generating new MNIST samples using an autoencoder.
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

dropout = tf.layers.dropout
conv2d = tf.layers.conv2d
max_pooling2d = tf.layers.max_pooling2d
dense = tf.layers.dense
flatten = tf.layers.flatten
relu = tf.nn.relu


# Dataset Parameters
num_classes = 10
img_shape = (28, 28, 1)
n_training_samples = 55000

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10
DROPOUT = 0.25

# Derived Parameters
_num_steps = EPOCHS * n_training_samples // BATCH_SIZE


def first_half(x_, is_training, dropout=DROPOUT):
    x = tf.reshape(x_, shape=(-1,) + img_shape)
    net = conv2d(x, 32, 3, activation=tf.nn.relu, name='conv1')
    net = max_pooling2d(net, 2, 2, name='pool1')
    net = conv2d(net, 64, 3, activation=tf.nn.relu, name='conv2')
    net = max_pooling2d(net, 2, 2, name='pool2')
    net = dropout(net, rate=dropout, training=is_training)
    net = flatten(net, name='flatten')
    net = dense(net, 128, activation=tf.nn.relu, name='fc1')
    net = dropout(net, rate=dropout, training=is_training)
    logits = dense(net, num_classes, name='fc2')
    return logits, x

unflatten = tf.reshape(output, [-1, 32, 256, 2])
def second_half(net, is_training, dropout=DROPOUT):
    net = dense(net, 128, activation=tf.nn.relu, name='ifc2')
    net = dropout(net, rate=dropout, training=is_training)
    net = dense(net, ???, activation=tf.nn.relu, name='ifc1')
    net = dropout(net, rate=dropout, training=is_training)
    net = tf.reshape(net, [-1, ???, ???, ???], name='unflatten')
    net = max_pooling2d(net, 2, 2, name='ipool2')

    ??? what should order of pool/conv/dropout layers be?
    return generated_images


def model_fcn(features, labels, mode, params):
    logits, x = first_half(features['images'],
                           mode == tf.estimator.ModeKeys.TRAIN)
    generated_images = second_half(tf.nn.softmax(logits),
                                   mode == tf.estimator.ModeKeys.TRAIN)
    y_hat = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
        loss, global_step=tf.train.get_global_step())

    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                              predictions=tf.argmax(y_hat, axis=1))

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y_hat,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc})
    return estim_specs


##########################################################
if __name__ == '__main__':
    # import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    # Build
    model = tf.estimator.Estimator(model_fcn,
                                   params={'learning_rate': LEARNING_RATE})

    # Train
    training_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)
    model.train(training_input_fn, steps=_num_steps)

    # Test
    testing_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images},
        y=mnist.test.labels,
        batch_size=BATCH_SIZE,
        shuffle=False)
    e = model.evaluate(testing_input_fn)
    print("Testing Accuracy:", e['accuracy'])
