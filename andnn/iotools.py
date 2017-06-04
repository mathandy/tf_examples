from __future__ import division, absolute_import, print_function

import os
import random

import h5py
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import imread
from sklearn.decomposition import IncrementalPCA
from tflearn.data_utils import build_hdf5_image_dataset

from andnn.utils import Timer


def k21hot(Y, k=None):
    if k is None:
        k = len(np.unique(Y))
    hot_labels = np.zeros((len(Y), k), dtype=Y.dtype)
    try:
        hot_labels[np.arange(len(Y)), Y % k] = 1
    except IndexError:
        hot_labels[np.arange(len(Y)), Y.astype(int) % k] = 1
    return hot_labels


def shuffle_together(list_of_arrays, permutation=None):
    m = len(list_of_arrays[0])
    assert all([len(x) == m for x in list_of_arrays[1:]])

    if permutation is None:
        permutation = list(range(m))
        random.shuffle(permutation)
    return [x[permutation] for x in list_of_arrays], permutation


def split_data(X, Y, validpart=0, testpart=0, shuffle=False):
    """Split data into training, validation, and test sets.  

    Assumes examples are indexed by first dimension.

    Args:
        X: any sliceable iterable
        Y: any sliceable iterable
        validpart: int or float proportion
        testpart: int or float proportion
        shuffle: bool

    Returns:
        (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    """
    assert validpart or testpart

    m = len(X)

    # shuffle data
    if shuffle:
        (X, Y), permutation = shuffle_together((X, Y))

    if 0 <= validpart < 1 and 0 <= testpart < 1:
        m_valid = int(validpart * m)
        m_test = int(testpart * m)
        m_train = len(Y) - m_valid - m_test
    else:
        m_valid = validpart
        m_test = testpart
        m_train = m - m_valid - m_test

    X_train = X[:m_train]
    Y_train = Y[:m_train]

    X_valid = X[m_train: m_train + m_valid]
    Y_valid = Y[m_train: m_train + m_valid]

    X_test = X[m_train + m_valid: len(X)]
    Y_test = Y[m_train + m_valid: len(Y)]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_h5(root_image_dir, image_shape, h5_dataset_name=None, mode='folder',
            categorical_labels=True, normalize=True, grayscale=False,
            files_extensions=('.png', '.jpg', '.jpeg'), chunks=False,
            ignore_existing=False):
    if h5_dataset_name is None:
        h5_dataset_name = os.path.join(root_image_dir,
                                       os.path.split(root_image_dir)[
                                           -1] + '.h5')
    if mode == 'unlabeled':
        unlabeled = True
    else:
        unlabeled = False
    if ignore_existing or not os.path.exists(h5_dataset_name):
        print("Creating hdf5 file...", end='')
        if unlabeled:
            mode = 'file'
            # create txt file listing images
            imlist_filename = os.path.join(root_image_dir, 'imlist.txt')
            with open(imlist_filename, 'w') as imlist:
                ims = os.listdir(root_image_dir)
                for im in ims:
                    line = os.path.join(root_image_dir, im) + ' 0\n'
                    imlist.write(line)
                root_image_dir = imlist_filename

        build_hdf5_image_dataset(root_image_dir,
                                 image_shape=image_shape,
                                 output_path=h5_dataset_name,
                                 mode=mode,
                                 categorical_labels=categorical_labels,
                                 normalize=normalize,
                                 grayscale=grayscale,
                                 files_extension=files_extensions,
                                 chunks=chunks)
        print('Done.')

    h5f = h5py.File(h5_dataset_name, 'r')
    X, Y = h5f['X'], h5f['Y']
    if unlabeled:
        return X
    return X, Y


# def whiten(X):
#     """PCA whitening."""
#     X -= np.mean(X, axis=0)  # zero-center
#     cov = np.dot(X.T, X) / X.shape[0]  # compute the covariance matrix
#     U, S, V = np.linalg.svd(cov)
#     X = np.dot(X, U)  # decorrelate the data
#     X /= np.sqrt(S + 1e-5)  # divide by the eigenvalues
#     return X


def incremental_whiten(X):
    def _whiten(A):
        ipca = IncrementalPCA(n_components=A.shape[1], whiten=True)
        return ipca.fit_transform(A)

    # center
    X -= np.mean(X, axis=0)

    # split channels and flatten
    m, w, h, d = X.shape
    assert m >= w * h
    channels = np.moveaxis(X, 3, 0).reshape(d, m, w * h)

    # whiten
    whitened_channels = np.stack([_whiten(c) for c in channels])
    # import pdb; pdb.set_trace()

    # put channels back in original shape and return
    return np.moveaxis(whitened_channels.reshape(d, m, w, h), 0, 3)


def image_preloader(image_directory, size, image_depth=3, label_type=False,
                    pixel_labels_lookup=None, num_classes=None,
                    exts=('.jpg', '.jpeg', '.png'), normalize=True,
                    shuffle=True, onehot=True, testpart=.0, validpart=.0,
                    whiten=False, ignore_existing=False,
                    storage_directory=None,
                    save_split_sets=True):
    """Image pre-loader (for use when entire dataset can be loaded in memory).

    This function is designed to load images store in directories in one of the
    following fashions (as determined by `label_type`):
        - subdirectory_labels images are in `image_directory` and 
        - 
        -

    All images are assumed to be in `image_directory`  or, if 
    `subdirectory_labels` is true then will assume all images are in a 
    subdirectory (of depth 1) and that subdirectory is a corresponding label.

    Args:
        image_directory (string): The root directory containing all images.
        size (array-like): The (width, height) images should be re-sized to.
        image_depth: the number of color channels in images.
        label_type (string): see description above.
        pixel_labels_lookup (func): function that returns an image given the
            filename and shape of an image.
        num_classes (int): Number of label classes.  Required only if 
            `label_type`=="pixel".
        exts (iterable): The set of acceptable file extensions to include.
        normalize: 
        shuffle: 
        onehot: 
        testpart: 
        validpart: 
        whiten: 
        ignore_existing: 
        storage_directory (string):  Should be provided if numpy arrays are to
            be saved (to save time if training images are ever be loaded again).
        save_split_sets (bool):

    Returns:
        (X_train, Y_train, X_valid, Y_valid, X_test, Y_test, class_names)
    """

    exists = os.path.exists
    join = os.path.join

    _label_types = ['subdirectory', 'pixel', None]
    # some parameter-checking
    if label_type == 'subdirectory':
        pass
    elif label_type == 'pixel':
        if pixel_labels_lookup is None:
            raise ValueError("If label_type=pixel, then pixel_labels_lookup "
                             "must be provided.")
        if num_classes is None:
            raise ValueError("If label_type=pixel, then num_classes "
                             "must be provided.")
    elif label_type is None:
        pass
    else:
        mes = "`label_type` must be one of the following options\n"
        mes += '\n'.join(_label_types)
        raise ValueError(mes)

    # some useful definitions
    s = '-{}x{}.npy'.format(*size)

    if image_depth in [None, 0, 1]:
        shape_of_input_images = size
    else:
        shape_of_input_images = (size[0], size[1], image_depth)

    if label_type == "pixel":
        shape_of_pixel_labels = (size[0], size[1], num_classes)

    def is_image(image_file_name):
        return os.path.splitext(image_file_name)[1] in exts

    if label_type == "subdirectory":
        class_names = [d for d in os.listdir('./' + image_directory)
                       if os.path.isdir(join(image_directory, d))]
    else:
        class_names = None

    if storage_directory:
        Yfile = join(storage_directory, 'Ydata' + s)
        Xfile = join(storage_directory, 'Xdata' + s)
        Xfile_white = join(storage_directory, 'Xdata-whitened-' + s)

    if storage_directory and not ignore_existing:

        if exists(Yfile) and \
              (exists(Xfile) or (whiten and exists(Xfile_white))):
            with Timer("loading target data from .npy files"):
                Y = np.load(Yfile)
            if whiten:
                if os.path.exists(Xfile_white):
                    with Timer("loading whitened data from .npy files"):
                        X = np.load(Xfile_white)
                else:
                    with Timer("loading unwhitened data from .npy files"):
                        X = np.load(Xfile)

                    with Timer('Whitening'):
                        X = incremental_whiten(X)

                    with Timer('Saving whitened data'):
                        np.save(Xfile_white, X)
            else:
                with Timer("loading input data from .npy file"):
                    X = np.load(Xfile)
        else:
            mes = 'No numpy file found.'
    else:
        mes = ''

        with Timer(mes + 'Loading data from image directories'):

            with Timer('Collecting image file names'):
                if label_type == "subdirectory":
                    image_files = []
                    Y = []
                    for k, d in enumerate(class_names):
                        with Timer(d, begin='\t'):
                            fd = join(image_directory, d)
                            image_files_d = [join(fd, fn)
                                             for fn in os.listdir(fd)
                                             if is_image(fn)]
                            Y += [k] * len(image_files_d)
                            image_files += image_files_d
                elif label_type == "pixel":
                    image_files = [join(image_directory, fn)
                                   for fn in os.listdir(image_directory)
                                   if is_image(fn)]
                else:  # Y is filenames
                    Y = image_files = [join(image_directory, fn)
                                       for fn in os.listdir(image_directory)
                                       if is_image(fn)]

            with Timer('\tLoading/Resizing images'):

                # initialize arrays
                X = np.empty((len(image_files), size[0], size[1], image_depth),
                             dtype=np.float32)
                if label_type == "subdirectory":
                    Y = np.array(Y).astype(np.float32)
                elif label_type == "pixel":
                    Y = np.empty(
                        (X.shape[0], X.shape[1], X.shape[2], num_classes),
                        dtype=np.float32)

                # resize and load images into arrays
                if label_type == "pixel":
                    for k, fn in enumerate(image_files):
                        imx = imread(fn).astype(np.float32)
                        imy = pixel_labels_lookup(fn, imx.shape)
                        if imx.shape != shape_of_input_images:
                            X[k] = imresize(imx, shape_of_input_images)
                            Y[k] = np.stack([imresize(imy[:, :, l], size)
                                             for l in range(num_classes)],
                                            axis=2)
                        else:
                            X[k] = imx
                            Y[k] = imy
                else:
                    for k, fn in enumerate(image_files):
                        im = imread(fn).astype(np.float32)
                        if im.shape != shape_of_input_images:
                            X[k, :, :, :] = imresize(im, shape_of_input_images)
                        else:
                            X[k, :, :, :] = im

        if shuffle:
            with Timer('Shuffling'):
                m = len(Y)
                permutation = list(range(m))
                random.shuffle(permutation)
                X = X[permutation]
                Y = Y[permutation]

        if onehot and label_type == "subdirectory":
            with Timer('Converting to 1-hot labels'):
                Y = k21hot(Y)

        if storage_directory:
            with Timer('Saving data (before any normalizing, whitening, and/or '
                       'splitting)'):
                np.save(Xfile, X)
                np.save(Yfile, Y)

        if whiten:
            with Timer('Whitening'):
                X = incremental_whiten(X)

            if storage_directory:
                with Timer('Saving whitening data'):
                    np.save(Xfile_white, X)

    if normalize and not whiten:
        with Timer('Normalizing data'):
            # X = (X - np.mean(X))/np.std(X)
            X -= np.mean(X, axis=0)
            X /= np.std(X, axis=0)

    if (testpart or validpart) and storage_directory:
        with Timer('Splitting data into fit/validate/test sets'):
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
                split_data(X, Y, validpart, testpart, shuffle=False)

        if save_split_sets:
            with Timer('Saving split datasets'):
                np.save(join(storage_directory, 'Xtrain' + s), X_train)
                np.save(join(storage_directory, 'Ytrain' + s), Y_train)
                if validpart:
                    np.save(join(storage_directory, 'Xvalid' + s), X_valid)
                    np.save(join(storage_directory, 'Yvalid' + s), Y_valid)
                if testpart:
                    np.save(join(storage_directory, 'Xtest' + s), X_test)
                    np.save(join(storage_directory, 'Ytest' + s), Y_test)
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, class_names
    else:
        return X, Y, None, None, None, None, class_names
