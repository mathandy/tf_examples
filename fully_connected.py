from __future__ import division, print_function, absolute_import
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from scipy.io import loadmat
from andnn import AnDNNClassifier
from andnn.iotools import k21hot, shuffle_together, split_data
from andnn.layers import fc_layer
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


def cewlogits(logits, y):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))


def get_batch(data, batch_size, step):
    offset = (step * batch_size) % (data.shape[0] - batch_size)
    return data[offset:(offset + batch_size)]


class Classifier:

    def __init__(self, model, example_shape, label_shape, final_activation=None,
                 learning_rate=1e-5, loss=cewlogits,
                 optimizer=tf.train.AdamOptimizer, session=tf.Session(),
                 debug=False):
        """Note: if loss expects logits, then `model` should output logits and 
        `final_activation` should be used."""
        self.example_shape = example_shape
        self.num_classes = label_shape

        self._X = tf.placeholder(tf.float32, [None] + example_shape)
        self._Y = tf.placeholder(tf.float32, [None] + label_shape)

        if final_activation is None:
            self._predictions = model(self._X)
            self._loss = loss(self._predictions, self._Y)
        else:
            self._logits = model(self._X)
            self._predictions = final_activation(self._logits)
            self._loss = loss(self._logits, self._Y)

        self._accuracy = accuracy(self._predictions, self._Y)
        self._number_correct = num_correct(self._predictions, self._Y)
        self._number_incorrect = num_incorrect(self._predictions, self._Y)

        # self._loss = tf.Print(self._loss,
        #                       data=[self._loss, self._accuracy,
        #                             self._number_correct,
        #                             self._number_incorrect],
        #                       message='loss, acc, cor, inc, = ',
        #                       first_n=100)

        self._train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self._loss)

        # initialize session
        self.session = tf.Session()
        debug = False
        if debug:
            from tensorflow.python import debug as tf_debug
            self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
            def always_true(*vargs):
                return True
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.session.add_tensor_filter("has_inf_or_nan", always_true)
    
    
    def fit(self, X_train, Y_train, batch_size, epochs, steps_per_report=500,
            X_valid=None, Y_valid=None):
        max_acc = 0
        self.session.run(tf.global_variables_initializer())
        step_accuracy = []
        step_loss = []
        try:
            for step in range(epochs * Y_train.shape[0]):
                # get batch ready for training step
                # feed_dict = {X: next(X_batch), Y: next(Y_batch)}
    
                X_batch = get_batch(X_train, batch_size, step)
                Y_batch = get_batch(Y_train, batch_size, step)
                feed_dict = {self._X: X_batch,
                             self._Y: Y_batch}
                fetches = [self._train_op, self._loss, self._accuracy]
    
                _, l, a = self.session.run(fetches, feed_dict)
                max_acc = max(max_acc, a)
                step_accuracy.append(a)
                step_loss.append(l)
                if (step % steps_per_report) == 0:
                    print("Step %04d | Loss = %.6f | a=%.2f" % (step, l, a))
                    if X_valid is not None:
                        self.validate(X_valid, Y_valid, batch_size, True)
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT")
        finally:
            print("max accuracy:", max_acc)
            step_plot([step_accuracy, step_loss],
                      ['step_accuracy', 'step_loss'])
    
    
    def validate(self, X, Y, batch_size, report=True):
        total_loss = 0
        total_correct = 0
        for X_batch, Y_batch in zip(batches(X, batch_size),
                                    batches(Y, batch_size)):
            feed_dict = {self._X: X_batch, self._Y: Y_batch}
            fetches = [self._loss, self._number_correct]

            l, c = self.session.run(fetches, feed_dict)

            total_loss += l*len(X_batch)/len(X)
            total_correct += c
        percent_correct = total_correct/len(X)
        print("Loss = %.6f | a=%.2f" % (total_loss, percent_correct))
        return total_loss, percent_correct


    def transform(self, X, batch_size, session=tf.Session()):
        for batch in batches(X, batch_size):
            _, l, a = self.session.run([self._predictions], {self._X: batch})
    
    
if __name__ == '__main__':
    usps_data = loadmat('usps/USPS.mat')
    X, Y = usps_data['fea'], usps_data['gnd']
    # X = X.reshape(-1, 16, 16)
    Y = k21hot(Y)

    Xtrain, Ytrain, Xvalid, Yvalid, _, _ = \
        split_data(X, Y, validpart=.2, testpart=0)
    # (X, Y), permutation = shuffle_together((X, Y))

    classifier = Classifier(multilayer_network,
                            final_activation=tf.nn.softmax,
                            example_shape=[256],
                            label_shape=[10],
                            learning_rate=1e-7,
                            debug=False)
    classifier.fit(X, Y, batch_size=50, epochs=50, steps_per_report=500,
                   X_valid=Xvalid, Y_valid=Yvalid)

    # if debug:
    #     print([v.name for v in tf.trainable_variables()])
