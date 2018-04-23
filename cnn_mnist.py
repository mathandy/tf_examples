""" 
A simple convnet classifier for MNIST

thanks: 
model parameters/structure taken from keras/examples
thanks also to aymericdamien/TensorFlow-Examples
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Training Parameters
learning_rate = 0.001
batch_size = 128
epochs = 10
dropout = 0.25


def network(img_shape, num_classes, is_training):
    x = tf.placeholder(tf.float32, (None,) + img_shape)
    conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2d = tf.layers.dropout(pool2, rate=dropout, training=is_training)
    fc1 = tf.layers.flatten(pool2d)
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
    fc2d = tf.layers.dropout(fc2, rate=dropout, training=is_training)
    logits = tf.layers.dense(fc2d, num_classes)
    return logits, x


def model_fcn(x_, y_, mode):
    y = tf.placeholder(tf.float32, (None, y_.shape[1]))
    logits, x = network(x_.shape[1:], y_.shape[1], mode=='train')
    y_hat = tf.nn.softmax(logits)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, 
                                                labels=y))

    update_step = tf.train.AdamOptimizer(learning_rate=learning_rate
        ).minimize(loss)

    _, acc = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), 
    					         predictions=tf.argmax(y_hat, axis=1))

    if mode=='train':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(epochs):
            # shuffle
            permutation = np.random.permutation(len(x_))
            x_, y_ = x_[permutation], y_[permutation]
            
            # train for an epoch
            for step in range(len(x_)//batch_size):
                x_batch = x_[step*batch_size: (step + 1)*batch_size]
                y_batch = y_[step*batch_size: (step + 1)*batch_size]

                _ = sess.run(fetches=[update_step], 
                             feed_dict={x: x_batch, y: y_batch})
            
            acc_, loss_ = sess.run(fetches=[acc, loss], 
            					   feed_dict={x: x_batch, y: y_batch})
            print("epoch: %s | loss = %s | acc = %s" % (epoch, loss_, acc_))
            # loss_ = sess.run(fetches=[loss], 
            # 			     feed_dict={x: x_batch, y: y_batch})
            # print("epoch: %s | loss = %s" % (epoch, loss_))
        return None, None
    elif mode=='test':
        return sess.run(fetches=[loss, acc], feed_dict={x: x_, y: y_})
    elif mode=='eval':
        return sess.run(fetches=[y_hat, p_hat], feed_dict={x: x_})
    else:
    	raise Exception("mode = %s is not an option." % mode)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

sess = tf.Session()
model_fcn(mnist.train.images.reshape((-1,28,28,1)), mnist.train.labels, 'train')
model_fcn(mnist.test.images.reshape((-1,28,28,1)), mnist.test.labels, 'test')
