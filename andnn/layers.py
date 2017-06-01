from __future__ import division, print_function, absolute_import
import tensorflow as tf


def conv_layer(input_tensor, diameter, in_dim, out_dim, act=tf.nn.relu,
               name=None):
    """Creates a convolutional layer with
    Args:
        input_tensor: A `Tensor`.
        diameter: An `int`, the width and also height of the filter.
        in_dim: An `int`, the number of input channels.
        out_dim: An `int`, the number of output channels.
        act: A `function`, the activation operation, defaults to tf.nn.relu.
        name: A `str`, the name for the operation defined by this function.
    """
    with tf.name_scope(name):
        filter_shape = (diameter, diameter, in_dim, out_dim)
        initial_weights = tf.truncated_normal(filter_shape, stddev=0.1)
        weights = tf.Variable(initial_weights, name='weights')

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='convolution')

        initial_biases = tf.constant(1.0, shape=[out_dim], dtype=tf.float32)
        biases = tf.Variable(initial_biases, name='biases')

        preactivations = tf.nn.bias_add(conv, biases, name='bias_addition')
        activations = tf.nn.relu(preactivations, name='activation')
    return activations, weights, biases


def fc_layer(in_tensor, in_dim, out_dim, act=tf.nn.relu, name=None):
    """Creates a fully-connected (ReLU by default) layer with
    Args:
        in_tensor: A `Tensor`.
        in_dim: An `int`, the number of input channels.
        out_dim: An `int`, the number of output channels.
        act: A `function`, the activation operation, defaults to tf.nn.relu.
        name: A `str`, the name for the operation defined by this function.
    """
    with tf.name_scope(name):
        initial_weights = tf.truncated_normal((in_dim, out_dim), stddev=0.1)
        weights = tf.Variable(initial_weights, name='weights')

        initial_biases = tf.constant(0.0, shape=[out_dim], dtype=tf.float32)
        biases = tf.Variable(initial_biases, name='biases')

        preactivations = tf.nn.bias_add(tf.matmul(in_tensor, weights), biases)
        if act is None:
            activations = preactivations
        else:
            activations = act(preactivations, name='activation')
    return activations, weights, biases
