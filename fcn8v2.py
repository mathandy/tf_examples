from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
from andnn.layers import conv3x3, conv

VGG_MEAN = [103.939, 116.779, 123.68]


def build(self, rgb, train=False, num_classes=20, random_init_fc8=False,
      debug=False):
    """
    Build the VGG model using loaded weights
    Parameters
    ----------
    rgb: image batch tensor
        Image in rgb shap. Scaled to Intervall [0, 255]
    train: bool
        Whether to build train or inference graph
    num_classes: int
        How many classes should be predicted (by fc8)
    random_init_fc8 : bool
        Whether to initialize fc8 layer randomly.
        Finetuning is required in this case.
    debug: bool
        Whether to print additional Debug Information.
    """

    # [From paper] Dropout is included where used in the original classifier
    # nets (however, training without it made little to no difference)
    keep_prob = 0.5

    # Convert RGB to BGR
    with tf.name_scope('Processing'):
        r, g, b = tf.split(rgb, 3, 3)
        bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], 3)

    # pooling arguments
    _ksize = [1, 2, 2, 1]
    _strides = [1, 2, 2, 1]

    # images --> conv1_1 --> conv1_2 --> pool1
    conv1_1, weights1, biases1 = conv3x3(bgr, 3, 3, 64, 'conv1_1')
    conv1_2, weights2, biases2 = conv3x3(conv1_1, 3, 64, 64, 'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')

    # pool1 --> conv2_1 --> conv2_2 --> pool2
    conv2_1, weights1, biases1 = conv3x3(pool1, 3, 64, 128, 'conv2_1')
    conv2_2, weights2, biases2 = conv3x3(conv2_1, 3, 128, 128, 'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')

    # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
    conv3_1, weights1, biases1 = conv3x3(pool2, 3, 128, 256, 'conv3_1')
    conv3_2, weights2, biases2 = conv3x3(conv3_1, 3, 256, 256, 'conv3_2')
    conv3_3, weights3, biases3 = conv3x3(conv3_2, 3, 256, 256, 'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')

    # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
    conv4_1, weights1, biases1 = conv3x3(pool3, 3, 256, 512, 'conv4_1')
    conv4_2, weights2, biases2 = conv3x3(conv4_1, 3, 512, 512, 'conv4_2')
    conv4_3, weights3, biases3 = conv3x3(conv4_2, 3, 512, 512, 'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')

    # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
    conv5_1, weights1, biases1 = conv3x3(pool4, 512, 512, 'conv5_1')
    conv5_2, weights2, biases2 = conv3x3(conv5_1, 512, 512, 'conv5_2')
    conv5_3, weights3, biases3 = conv3x3(conv5_2, 512, 512, 'conv5_3')
    pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')

    if name == 'fc6':
        filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
    elif name == 'score_fr':
        name = 'fc8'  # Name of score_fr layer in VGG Model
        filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                          num_classes=num_classes)
    else:
        filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

    # BIG QUESTION: why good idea to just reshape weights from vgg???
    # do they do this in paper?

    fc6 = conv(pool5, [7, 7, 512, 4096], "fc6")
    if train:
        fc6 = tf.nn.dropout(self.fc6, keep_prob)

    fc7 = conv(pool5, [1, 1, 4096, 4096], "fc7")
    if train:
        self.fc7 = tf.nn.dropout(self.fc7, keep_prob)

    if random_init_fc8:
        score_fr = self._score_layer(self.fc7, "score_fr", num_classes)
    else:
        score_fr = conv(fc7, [1, 1, 4096, 1000], "fc8", act=None)

    self.pred = tf.argmax(self.score_fr, dimension=3)

    self.upscore2 = self._upscore_layer(self.score_fr,
                                        shape=tf.shape(self.pool4),
                                        num_classes=num_classes,
                                        debug=debug, name='upscore2',
                                        ksize=4, stride=2)
    self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                         num_classes=num_classes)
    self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

    self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                        shape=tf.shape(self.pool3),
                                        num_classes=num_classes,
                                        debug=debug, name='upscore4',
                                        ksize=4, stride=2)
    self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                         num_classes=num_classes)
    self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

    self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                         shape=tf.shape(bgr),
                                         num_classes=num_classes,
                                         debug=debug, name='upscore32',
                                         ksize=16, stride=8)

    self.pred_up = tf.argmax(self.upscore32, dimension=3)
    return self.upscore32  # maybe they train with something else


def _fc_layer(self, bottom, name, num_classes=None,
              relu=True, debug=False):
    with tf.variable_scope(name) as scope:
        shape = bottom.get_shape().as_list()

        if name == 'fc6':
            filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
        elif name == 'score_fr':
            name = 'fc8'  # Name of score_fr layer in VGG Model
            filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                              num_classes=num_classes)
        else:
            filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

        self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = self.get_bias(name, num_classes=num_classes)
        bias = tf.nn.bias_add(conv, conv_biases)

        if relu:
            bias = tf.nn.relu(bias)
        _activation_summary(bias)

        if debug:
            bias = tf.Print(bias, [tf.shape(bias)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return bias

def _score_layer(self, bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input)**0.5
        elif name == "score_pool4":
            stddev = 0.001
        elif name == "score_pool3":
            stddev = 0.0001
        # Apply convolution
        w_decay = self.wd

        weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                   decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = self._bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias

def _upscore_layer(self, bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = self.get_deconv_filter(f_shape)
        self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    _activation_summary(deconv)
    return deconv

def get_deconv_filter(self, f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def get_conv_filter(self, name):
    init = tf.constant_initializer(value=self.data_dict[name][0],
                                   dtype=tf.float32)
    shape = self.data_dict[name][0].shape
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))
    var = tf.get_variable(name="filter", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def get_bias(self, name, num_classes=None):
    bias_wights = self.data_dict[name][1]
    shape = self.data_dict[name][1].shape
    if name == 'fc8':
        bias_wights = self._bias_reshape(bias_wights, shape[0],
                                         num_classes)
        shape = [num_classes]
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="biases", initializer=init, shape=shape)
    _variable_summaries(var)
    return var

def get_fc_weight(self, name):
    init = tf.constant_initializer(value=self.data_dict[name][0],
                                   dtype=tf.float32)
    shape = self.data_dict[name][0].shape
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def _bias_reshape(self, bweight, num_orig, num_new):
    """ Build bias weights for filter produces with `_summary_reshape`

    """
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight

def _summary_reshape(self, fweight, shape, num_new):
    """ Produce weights for a reduced fully-connected layer.

    FC8 of VGG produces 1000 classes. Most semantic segmentation
    task require much less classes. This reshapes the original weights
    to be used in a fully-convolutional layer which produces num_new
    classes. To archive this the average (mean) of n adjanced classes is
    taken.

    Consider reordering fweight, to perserve semantic meaning of the
    weights.

    Args:
      fweight: original weights
      shape: shape of the desired fully-convolutional layer
      num_new: number of new classes


    Returns:
      Filter weights for `num_new` classes.
    """
    num_orig = shape[3]
    shape[3] = num_new
    assert(num_new < num_orig)
    n_averaged_elements = num_orig//num_new
    avg_fweight = np.zeros(shape)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        avg_fweight[:, :, :, avg_idx] = np.mean(
            fweight[:, :, :, start_idx:end_idx], axis=3)
    return avg_fweight

def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _add_wd_and_summary(self, var, wd, collection_name=None):
    if collection_name is None:
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _bias_variable(self, shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='biases', shape=shape,
                          initializer=initializer)
    _variable_summaries(var)
    return var

def get_fc_weight_reshape(self, name, shape, num_classes=None):
    print('Layer name: %s' % name)
    print('Layer shape: %s' % shape)
    weights = self.data_dict[name][0]
    weights = weights.reshape(shape)
    if num_classes is not None:
        weights = self._summary_reshape(weights, shape,
                                        num_new=num_classes)
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)


# if __name__ == '__main__':
#
#     if True:
#         from scipy.io import loadmat
#         from andnn import AnDNNClassifier
#         from andnn.iotools import k2pixelwise, split_data
#         from andnn.losses import ce_wlogits
#
#         usps_data = loadmat('usps/USPS.mat')
#         X, Y = usps_data['fea'], usps_data['gnd']
#         X = X.reshape(-1, 16, 16)
#
#         X = np.stack((X, X, X), axis=3)                # pretend USPS is colored
#         # print(Y.shape)
#         # print(Y[:3])
#         Y = k2pixelwise(Y, X.shape[1:3], onehot=True)  # and has pixel-wise labels
#         # print(Y.shape)
#         # for i in range(3):
#         #     print("i=",i,":",Y[i].argmax())
#         #     print(Y[i,1,1,:])
#
#         Xtrain, Ytrain, Xvalid, Yvalid, _, _ = \
#             split_data(X, Y, validpart=.2, testpart=0)
#
#         vgg_weights = '/home/andy/Desktop/KittiSeg/data/vgg16.npy'
#         # vgg_weights = np.load(vgg_weights, encoding='latin1').item()
#         fcn8 = lambda rgb: FCN8VGG(vgg_weights).build(rgb,
#                                                       train=True,
#                                                       num_classes=Y.shape[-1],
#                                                       random_init_fc8=False,
#                                                       debug=False)
#         classifier = AnDNNClassifier(fcn8,
#                                      final_activation=tf.nn.softmax,
#                                      example_shape=X.shape[1:],
#                                      label_shape=Y.shape[1:],
#                                      debug=False)
#         # classifier.load_weights(vgg_weights)
#         classifier.fit(X, Y, batch_size=1, epochs=20,
#                        loss=ce_wlogits,
#                        loss_kwargs={},
#                        optimizer=tf.train.AdamOptimizer,
#                        optimizer_kwargs={'learning_rate': 1e-4},
#                        steps_per_report=500,
#                        X_valid=Xvalid,
#                        Y_valid=Yvalid,
#                        validation_batch_size=100)

    # if debug:
    #     print([v.name for v in tf.trainable_variables()])
if __name__ == '__main__':
    import os
    import scipy as scp
    import scipy.misc

    import numpy as np
    import logging
    import tensorflow as tf
    import sys

    from andnn.utils import color_image, Timer
    from andnn import AnDNNClassifier

    # parameters
    test_image = scp.misc.imread("tabby_cat.png")
    num_classes = 20  # really?

    vgg_weights = 'models/vgg16.npy'
    if not os.path.exists(vgg_weights):
        vgg_weights = os.path.expanduser('~/Desktop/vgg16.npy')

    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
    #                     level=logging.INFO,
    #                     stream=sys.stdout)



    fcn = FCN8VGG(vgg_weights)
    def model(rgb):
        return fcn.build(rgb,
                         train=False,
                         num_classes=num_classes,
                         random_init_fc8=False,
                         debug=False)
    classifier = AnDNNClassifier(model,
                                 final_activation=tf.nn.softmax,
                                 example_shape=test_image.shape,
                                 label_shape=test_image.shape[:-1] + (20,),
                                 debug=False)

    # init = tf.global_variables_initializer()
    # classifier.session.run(init)

    classifier._initialize()
    with Timer('Running the Network'):

        fetches = [fcn.pred, fcn.pred_up]
        feed_dict = {classifier._X: np.expand_dims(test_image, 0)}
        down, up = classifier.session.run(fetches, feed_dict=feed_dict)

        down_color = color_image(down[0])
        up_color = color_image(up[0])

        scp.misc.imsave('fcn8_downsampled.png', down_color)
        scp.misc.imsave('fcn8_upsampled.png', up_color)
