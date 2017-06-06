from __future__ import division, print_function, absolute_import
from six.moves import zip
import tensorflow as tf
from andnn.losses import ce_wlogits
from andnn.utils import step_plot, accuracy, num_correct, num_incorrect, batches


def get_batch(data, batch_size, step):
    offset = (step * batch_size) % (data.shape[0] - batch_size)
    return data[offset:(offset + batch_size)]


class AnDNNClassifier:
    def __init__(self, model, example_shape, label_shape, final_activation=None,
                 session=None, debug=False):
        """Note: if loss expects logits, then `model` should output logits and 
        `final_activation` should be used."""
        self.model = model
        self.example_shape = tuple(example_shape)
        self.label_shape = tuple(label_shape)
        self.final_activation = final_activation
        self.session = session
        self.debug = debug

        self._X = tf.placeholder(tf.float32, (None,) + self.example_shape)
        self._Y = tf.placeholder(tf.float32, (None,) + self.label_shape)

    def _initialize_session(self):
        # initialize session
        if self.session is None:
            self.session = tf.Session()
        if self.debug:
            from tensorflow.python import debug as tf_debug
            self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)

            def always_true(*vargs):
                return True

            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.session.add_tensor_filter("has_inf_or_nan", always_true)

    def _build_learning_mechanism(self, loss, loss_kwargs,
                                  optimizer, optimizer_kwargs):
        if self.final_activation is None:
            self._predictions = self.model(self._X)
            self._loss = loss(self._predictions, self._Y, **loss_kwargs)
        else:
            self._logits = self.model(self._X)
            self._predictions = self.final_activation(self._logits)
            self._loss = loss(self._logits, self._Y)

        self._accuracy = accuracy(self._predictions, self._Y)
        self._number_correct = num_correct(self._predictions, self._Y)
        self._number_incorrect = num_incorrect(self._predictions, self._Y)

        self._loss = tf.Print(self._loss,
                              data=[self._loss, self._accuracy,
                                    self._number_correct,
                                    self._number_incorrect],
                              message='\nloss, acc, cor, inc, = ',
                              first_n=100)

        self._train_op = optimizer(**optimizer_kwargs).minimize(self._loss)

        self._initialize_session()

    def fit(self, X_train, Y_train, batch_size, epochs, loss=ce_wlogits,
            loss_kwargs=None, optimizer=tf.train.AdamOptimizer,
            optimizer_kwargs={'learning_rate': 1e-5}, steps_per_report=500,
            X_valid=None, Y_valid=None):
        self._build_learning_mechanism(loss=loss,
                                       loss_kwargs=loss_kwargs,
                                       optimizer=optimizer,
                                       optimizer_kwargs=optimizer_kwargs)
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
                    if X_valid is not None:
                        vl, va = self.validate(X_valid, Y_valid,
                                               batch_size, False)
                        print("train | val "
                              ":: {: 6.2%} | {: 6.2%} "
                              ":: {: 5.4G} | {: 5.4G} "
                              ":: step {} / {}"
                              "".format(a, va, l, vl, step,
                                        epochs * Y_train.shape[0]))

                    else:
                        print("train | val "
                              ":: {: 6.2%} | N/A "
                              ":: {: 5.4G} | N/A "
                              ":: step {} / {}"
                              "".format(a, l, step,
                                        epochs * Y_train.shape[0]))

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
        if report:
            print("Validation: Loss = {:0.4G} | a={:06.2%}"
                  "".format(total_loss, percent_correct))
        return total_loss, percent_correct

    def transform(self, X, batch_size, session=tf.Session()):
        for batch in batches(X, batch_size):
            _, l, a = self.session.run([self._predictions], {self._X: batch})

    def close(self):
        self.session.close()
