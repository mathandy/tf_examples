from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import os



class AnDNN:
    def __init__(self, activations, parameters, weights_file=None, session=None,
                 tensorboard_dir='/tmp/tflogs'):
        """
        
        Parameters
        ----------
        model_dict: dict
            A dictionary of tf operators that are defined in the model.  The 
            purpose being to give a simple accesible name space for the fetches
            and result_handling functions in AnDNN().fit()
        steps_per_report
        steps_per_save
        weights_file
        session
        tensorboard_dir
        """

        self.tensorboard_dir = tensorboard_dir
        self.input_images = tf.placeholder(tf.float32, (None, 224, 224, 3))
        # (self.parameters, self.training_step_update_op, self.summary_op,
        #  self.training_loss_op, self.validation_loss_op) = model
        self.model_dict = model_dict
        self.parameters = model_dict['parameters']

        # maybe initialize a session
        if session is None:
            self._sess = tf.Session()
        else:
            self._sess = session

        # load or initialize weights
        self._sess.run(tf.global_variables_initializer())
        if weights_file is not None:
            self._sess.run(tf.global_variables_initializer())
            self.load_weights(weights_file)

        # construct saver object
        self._saver = tf.train.Saver(self.parameters)

    def load_weights(self, weight_file, np_load_kwargs={}):
        """Loads a dictionary of weights."""
        weights = np.load(weight_file, **np_load_kwargs)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            self._sess.run(self.parameters[i].assign(weights[k]))

    # def save_weights(self, weight_file, np_savez_kwargs):
    #     keys = sorted(self.parameters)
    #     weights = np.savez(weight_file, self.parameters)
    #
    #     for i, k in enumerate(keys):
    #         self._sess.run(self.parameters[i].assign(weights[k]))

    def create_checkpoint(self, step, checkpoint_name=None):
        self._saver.save(self._sess, checkpoint_name, global_step=step)

    def _step_forward(self):

    def _default_step_fcn(self, feed_dict, step):
        steps_per_save = 500
        steps_per_report = 10
        md = self.model_dict

        if step > 0 and step % steps_per_save == 0:
            # fetches = [md['step_forward'],
            #            md['update_summary'],
            #            md['training_loss'],
            #            md['validation_loss']]
            fetches = [md['step_forward'],
                       md['update_summary'],
                       md['loss']]
            _, summary, training_loss = \
                self._sess.run(fetches, feed_dict=feed_dict)
            self.create_checkpoint(step)
            print('Step: {} | Loss: {} | Validation Loss: {}%'
                  ''.format(step, training_loss, None))

        elif step > 0 and step % steps_per_report == 0:
            # fetches = [md['step_forward'],
            #            md['update_summary'],
            #            md['training_loss'],
            #            md['validation_loss']]
            fetches = [md['step_forward'],
                       md['update_summary'],
                       md['loss']]
            _, summary, training_loss = \
                self._sess.run(fetches, feed_dict=feed_dict)
            print('Step: {} | Loss: {} | Validation Loss: {}%'
                  ''.format(step, training_loss, None))
        else:
            fetches = [md['step_forward'],
                       md['update_summary'],
                       md['loss']]
            _, summary, training_loss = \
                self._sess.run(fetches, feed_dict=feed_dict)

        return summary

    def fit(self, X, Y, batch_size, valid=0.0, run_id='unnamed',
            epochs=10, steps_per_save=500, steps_per_report=1, step_fcn=None):

        if step_fcn is None:
            step_fcn = self._default_step_fcn

        file_writer = tf.summary.FileWriter(self.tensorboard_dir + run_id,
                                            self._sess.graph)

        # def batch(data_, batch_size_, epochs_):
        #     for step in range(epochs_ * Y.shape[0]):
        #         offset = (step * batch_size_) % (data_.shape[0]-batch_size_)
        #         yield data_[offset:(offset + batch_size_), :]
        #
        # X_batch = batch(X, batch_size, epochs)
        # Y_batch = batch(Y, batch_size, epochs)

        def get_batch(data_, batch_size_, step_):
            offset = (step_ * batch_size_) % (data_.shape[0]-batch_size_)
            return data_[offset:(offset + batch_size_), :]

        # Training Step
        for step in range(epochs * Y.shape[0]):
                # get batch ready for training step
                # feed_dict = {X: next(X_batch), Y: next(Y_batch)}

                X_batch = get_batch(X, batch_size, step)
                Y_batch = get_batch(Y, batch_size, step)
                feed_dict = {self.model_dict['X']: X_batch,
                             self.model_dict['Y']: Y_batch}

                tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                summary = step_fcn(feed_dict, step)

                file_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                if summary:
                    file_writer.add_summary(summary, step)

    def get_output(self, images, auto_resize=True):
        """"Takes in a list of images and returns softmax probabilities."""
        if auto_resize:
            images_ = [imresize(im, (224, 224)) for im in images]
        else:
            images_ = images
        feed_dict = {self.input_images: images_}
        return self._sess.run(self.output, feed_dict)[0]

    def get_activations(self, images, auto_resize=True):
        """"Takes in a list of images and returns the activation dictionary."""
        if auto_resize:
            images_ = np.array([imresize(im, (224, 224)) for im in images])
        else:
            images_ = np.array(images)
        feed_dict = {self.input_images: images_}
        return self._sess.run(self.activations, feed_dict)[0]


# if __name__ == '__main__':
#     image_dir = 'data'
#     image_size = (224, 224)
#
#     # Pre-load data
#     X, Y = load_h5(image_dir, image_size)
#
#     # Initialize VGG16
#     vgg = VGG16('vgg16_weights.npz')
#
#     # Run images through network, return softmax probabilities
#     class_probabilities = vgg.get_output(input_images)
#     print(class_probabilities.shape)
#
#     # # Get Class Names
#     # def get_class_names():
#     #     with open('ImageNet_Classes.txt') as names_file:
#     #         return [l.replace('\n', '') for l in names_file]
#     # class_names = get_class_names()
