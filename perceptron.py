from __future__ import division, print_function, absolute_import
from scipy.io import loadmat
from andnn.iotools import split_data, k21hot

usps_data = loadmat('usps.mat')
X, Y = usps_data['fea'], usps_data['gnd']
X = X.reshape(-1, 16, 16)
Y = k21hot(Y)

Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = \
    split_data(X, Y, validpart=.2, testpart=.2)







