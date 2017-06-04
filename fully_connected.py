from __future__ import division, print_function, absolute_import
import tensorflow as tf
from scipy.io import loadmat
from andnn.iotools import k21hot, shuffle_together
from andnn.layers import fc_layer
from tensorflow.contrib import layers
from andnn.utils import step_plot, accuracy, num_correct, num_incorrect


# def multilayer_network(x):
#     fc1, _, _ = fc_layer(x, 256, 256, act=tf.nn.relu, name='fc1')
#     fc2, _, _ = fc_layer(fc1, 256, 256, act=tf.nn.relu, name='fc2')
#     fc3pre, _, _ = fc_layer(fc2, 256, 10, act=None, name='fc3pre')
#     return fc3pre


def multilayer_network(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
    return out


x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 10])
pred = multilayer_network(x)
acc = accuracy(pred, y)
num_corr = num_correct(pred, y)
num_incorr =num_incorrect(pred, y)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
loss = tf.Print(loss, data=[loss, acc, num_corr, num_incorr], message='loss, acc, cor, inc, = ', first_n=100)
train_op = tf.train.AdamOptimizer(learning_rate=0.0000001).minimize(loss)


def get_batch(data_, batch_size_, step_):
    offset = (step_ * batch_size_) % (data_.shape[0] - batch_size_)
    return data_[offset:(offset + batch_size_), :]


def fit(X, Y, batch_size, epochs, session=tf.Session(), steps_per_report=500):
    max_acc = 0
    session.run(tf.global_variables_initializer())
    step_accuracy = []
    step_loss = []
    try:
        for step in range(epochs * Y.shape[0]):
            # get batch ready for training step
            # feed_dict = {X: next(X_batch), Y: next(Y_batch)}

            X_batch = get_batch(X, batch_size, step)
            Y_batch = get_batch(Y, batch_size, step)
            feed_dict = {x: X_batch,
                         y: Y_batch}

            _, l, a = session.run([train_op, loss, acc], feed_dict)
            max_acc = max(max_acc, a)
            step_accuracy.append(a)
            step_loss.append(l)
            if (step % steps_per_report) == 0:
                print("Step %04d | Loss = %.6f | a=%.2f" % (step, l, a))
    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT")
    finally:
        print("max accuracy:", max_acc)
        step_plot([step_accuracy, step_loss], ['step_accuracy', 'step_loss'])


def transform(X, batch_size, session=tf.Session()):



if __name__ == '__main__':
    usps_data = loadmat('usps/USPS.mat')
    X, Y = usps_data['fea'], usps_data['gnd']
    # X = X.reshape(-1, 16, 16)
    Y = k21hot(Y)

    (X, Y), permutation = shuffle_together((X, Y))

    sess = tf.Session()
    debug = False
    if debug:
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        def always_true(*vargs):
            return True
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("has_inf_or_nan", always_true)

    sess = fit(X, Y, batch_size=50, epochs=50, session=sess)
    if debug:
        print([v.name for v in tf.trainable_variables()])
