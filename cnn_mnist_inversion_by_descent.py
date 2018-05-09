""" 
Unfinished.
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

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
USE_DROPOUT = False  # try true also!!!!!!!!!!!!!!!!!!
max_steps = 1000
step_per_report = 100


_network_parameter_names = ['conv1/bias', 'conv1/kernel',
                            'conv2/bias', 'conv2/kernel',
                            'fc1/bias', 'fc1/kernel',
                            'fc2/bias', 'fc2/kernel']


def reparameterized_network(is_training, params):
    x = tf.Variable(tf.truncated_normal((10,) + img_shape), name='gen_images')
    conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, name='conv1',
     trainable=False,
     kernel_initializer=tf.constant_initializer(params['conv1/kernel']),
     bias_initializer=tf.constant_initializer(params['conv1/bias']))
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, name='conv2',
     trainable=False,
     kernel_initializer=tf.constant_initializer(params['conv2/kernel']),
     bias_initializer=tf.constant_initializer(params['conv2/bias']))
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
    pool2d = tf.layers.dropout(pool2, rate=DROPOUT, training=is_training, name='pool2_dropout')
    pool2df = tf.layers.flatten(pool2d, name='pool2d_flattened')
    fc1 = tf.layers.dense(pool2df, 128, activation=tf.nn.relu, name='fc1',
     trainable=False,
     kernel_initializer=tf.constant_initializer(params['fc1/kernel']),
     bias_initializer=tf.constant_initializer(params['fc1/bias']))
    fc1d = tf.layers.dropout(fc1, rate=DROPOUT, training=is_training, name='fc1_dropout')
    logits = tf.layers.dense(fc1d, num_classes, name='fc2',
     trainable=False,
     kernel_initializer=tf.constant_initializer(params['fc2/kernel']),
     bias_initializer=tf.constant_initializer(params['fc2/bias']))
    return logits, x


# def reparameterized_model_fcn(features, labels, mode, params=None):
#     logits, x = reparameterized_network(USE_DROPOUT, params)
#     y_hat = tf.nn.softmax(logits)
#
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#         logits=logits,
#         labels=labels))
#
#     train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
#         loss, global_step=tf.train.get_global_step())
#
#     acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
#                               predictions=tf.argmax(y_hat, axis=1))
#
#     estim_specs = tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=y_hat,
#         loss=loss,
#         train_op=train_op,
#         eval_metric_ops={'accuracy': acc})
#     return estim_specs


def reparameterized_model_fcn(params, labels):
    logits, x = reparameterized_network(USE_DROPOUT, params)
    y_hat = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss, global_step=tf.train.get_global_step())

    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                              predictions=tf.argmax(y_hat, axis=1))

    return acc, loss, train_op, y_hat, x

##########################################################


# Train
def pretrain_parameters():
    # import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    from cnn_mnist_using_tf_estimator import model_fcn
    model = tf.estimator.Estimator(model_fcn,
                                   params={'learning_rate': TRAINING_LEARNING_RATE})

    # Train
    training_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, 
        y=mnist.train.labels,
        batch_size=TRAINING_EPOCHS,
        num_epochs=TRAINING_BATCH_SIZE,
        shuffle=True)
    model.train(training_input_fn, steps=1000)

    # Test
    testing_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, 
        y=mnist.test.labels,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=False)
    e = model.evaluate(testing_input_fn)
    print("Testing Accuracy:", e['accuracy'])

    params = {name: model.get_variable_value(name) 
                for name in _network_parameter_names}
    return params


# def generate_examples(pretrained_params, labels):
#     # build reparameterized model with pretrained parameters
#     rmodel = tf.estimator.Estimator(reparameterized_model_fcn,
#                                     params=pretrained_params)
#
#     gen_input_fcn = tf.estimator.inputs.numpy_input_fn(
#         x={'images', np.zeros((100,100,3))},
#         y=labels,
#         batch_size=len(labels),
#         shuffle=False,
#     )
#
#     rmodel.train(gen_input_fcn, steps=NUM_STEPS)
#     return rmodel.get_variable_value('gen_images')


def generate_examples(pretrained_params, labels):
    with tf.Session() as sess:

        acc, loss, train_op, y_hat, x = \
            reparameterized_model_fcn(pretrained_params, labels)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for step in range(max_steps):
            _ = sess.run(fetches=[train_op])

            if not (step % step_per_report):
                acc_, loss_ = sess.run(fetches=[acc, loss])
                print("step: %s | loss = %s | acc = %s" % (step, loss_, acc_))


        return sess.run(x)


if __name__ == '__main__':
    try:
        pretrained_parameters = np.load('cnn_mnist_parameters.npz')
    except FileNotFoundError:
        pretrained_parameters = pretrain_parameters()
        np.savez('cnn_mnist_parameters', **pretrained_parameters)

    pretrained_parameters = dict(pretrained_parameters)
    generated = generate_examples(pretrained_parameters,
                                  np.identity(num_classes))




    # from cv2 import imshow
    for k, ex in enumerate(generated):
        # imshow('generated %s' % k, ex)
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        imgplot = plt.imshow(ex)