""" 
A simple convnet classifier for MNIST using a tf.estimator

thanks: 
model parameters/structure taken from keras/examples
thanks also to aymericdamien/TensorFlow-Examples
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# dataset parameters
num_classes = 10
img_shape = (28,28)
n_training_samples = 55000

# Training Parameters
learning_rate = 0.001
batch_size = 128
epochs = 10
num_steps = epochs*n_training_samples//batch_size
dropout = 0.25


def network(x_, is_training):
    x = tf.reshape(x_, shape=(-1,) + img_shape + (1,))
    conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2d = tf.layers.dropout(pool2, rate=dropout, training=is_training)
    pool2df = tf.layers.flatten(pool2d)
    fc1 = tf.layers.dense(pool2df, 128, activation=tf.nn.relu)
    fc1d = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    logits = tf.layers.dense(fc1d, num_classes)
    return logits, x


def model_fcn(features, labels, mode):
    logits, x = network(features['images'], mode==tf.estimator.ModeKeys.TRAIN)
    y_hat = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, 
        labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
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
# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# Build
model = tf.estimator.Estimator(model_fcn)

# Train
training_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, 
    y=mnist.train.labels,
    batch_size=batch_size, 
    num_epochs=epochs, 
    shuffle=True)
model.train(training_input_fn, steps=num_steps)

# Test
testing_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, 
    y=mnist.test.labels,
    batch_size=batch_size, 
    shuffle=False)
e = model.evaluate(testing_input_fn)
print("Testing Accuracy:", e['accuracy'])
