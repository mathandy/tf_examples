from __future__ import division, print_function, absolute_import
import tensorflow as tf
from scipy.io import loadmat
from andnn.iotools import k21hot, shuffle_together
from andnn.layers import fc_layer


# https://wookayin.github.io/tensorflow-talk-debugging/#15
def multilayer_perceptron(x):

    fc1, _, _ = fc_layer(x, 256, 256, act=tf.nn.relu, name='fc1')
    fc2, _, _ = fc_layer(fc1, 256, 256, act=tf.nn.relu, name='fc2')
    fc3pre, _, _ = fc_layer(fc2, 256, 10, act=None, name='fc3pre')
    return fc3pre


x = tf.placeholder(tf.float32, [None, 256])
y = tf.placeholder(tf.float32, [None, 10])
pred = multilayer_perceptron(x)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# loss = tf.Print(loss, data=[loss], message='\n\nloss shape = ')
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


def accuracy(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return 100.0 * tf.reduce_mean(tf.cast(is_correct, "float"))


def num_correct(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(is_correct, "float"))


def num_incorrect(predictions, labels):
    is_incorrect = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(is_incorrect, "float"))

acc = accuracy(pred, y)
num_corr = num_correct(pred, y)
num_incorr =num_incorrect(pred, y)

loss = tf.Print(loss, data=[acc, num_corr, num_incorr], message='acc, cor, inc, = ')

def get_batch(data_, batch_size_, step_):
    offset = (step_ * batch_size_) % (data_.shape[0] - batch_size_)
    return data_[offset:(offset + batch_size_), :]


def train(X, Y, epochs, session=tf.Session(), steps_per_report=1):
    batch_size = 200
    session.run(tf.global_variables_initializer())
    for step in range(epochs * Y.shape[0]):
        # get batch ready for training step
        # feed_dict = {X: next(X_batch), Y: next(Y_batch)}

        X_batch = get_batch(X, batch_size, step)
        Y_batch = get_batch(Y, batch_size, step)
        feed_dict = {x: X_batch,
                     y: Y_batch}

        _, l = session.run([train_op, loss], feed_dict)
        if (step % steps_per_report) == 0:
            print("Step %04d, Loss = %.6f" % (step, l))


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

    sess = train(X, Y, epochs=2, session=sess)
    if debug:
        print([v.name for v in tf.trainable_variables()])
