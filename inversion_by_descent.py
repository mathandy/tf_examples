"""This is an unfinished experiment I just began working on."""

import tensorflow as tf
import matplotlib.pyplot as plt
# plt.ion()
import numpy as np
from polynomial_regression import poly_fit, gen_fake_data_poly


NUM_SAMPLES = 2**13  # number of fake training examples
DEGREE = 3  # degree of polynomial 
NOISE_FACTOR = 10  # how much noise in training examples 



if __name__=='__main__':
    x_train, y_train, p = gen_fake_data_poly(NUM_SAMPLES, DEGREE, NOISE_FACTOR)
    xb, yh, c = poly_fit(x_train, y_train, 
                      batch_size=32, epochs=10, learning_rate = 0.01, 
                      deg=DEGREE)

    print('coeffs:', c.ravel())

    # plot results
    xb, yh = zip(*sorted(zip(xb,yh)))
    plt.plot(xb, yh, 'ro', x_train, y_train, 'b--')
    plt.show()