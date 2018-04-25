"""This is a simple case demonstrating the idea that after training a 
model, $y=f_{\theta^*}(x)$, i.e. finding,
 $\theta^* = \min_\theta\{y=f_\theta(x)\}$, 
 it's possible to invert the model at a point (non-uniquely) by finding, 
 $x^* = \min_x\{y=f_\theta(x)\}$.

 In other words, this is Newton's method on a polynomial.

 There are immediate disadvantages of this method:
 1. the model cannot generalize beyond those y's given as we're 
 the values we're learning are exactly the values we desire to generate. 
 2. batched gradient descent isn't possible in its usual form.
 3. the only stochastic component is the initial guesses for x."""

import tensorflow as tf
import matplotlib.pyplot as plt
# plt.ion()
import numpy as np
from polynomial_regression import poly_fit, gen_fake_data_poly


NUM_SAMPLES = 2**13  # number of fake training examples
DEGREE = 3  # degree of polynomial 
LEARNING_RATE = 0.01
STEPS = 100


def inverse_poly_fit(coeffs, y_, steps, learning_rate):
    """Given the coefficients, [c_0, ..., c_n], of a polynomial,
    p(x) = c_n*x^n + ... + c_0, and values y, uses gradient decent to 
    find corresponding x coordinates such that p(x) = y."""

    c = tf.constant(coeffs)
    x = tf.Variable(tf.truncated_normal(y_.shape))
    y = tf.constant(y_)

    # find p(x) and loss
    X = [tf.ones(tf.shape(x)), x] + [x**k for k in range(2, len(coeffs+1))]
    X = tf.stack(X, axis=1)
    y_hat = tf.squeeze(tf.matmul(X, c))
    loss = tf.reduce_sum((y - y_hat)**2)/np.prod(np.shape(y_))

    # "Train"
    update_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(steps):

            loss_, _ = sess.run(fetches=[loss, update_step])        
            print("step: %s | loss = %s" % (step, loss_))

        return sess.run(x)


if __name__=='__main__':
    # generate a polynomial to use for this example
    x_train, y_train, p = gen_fake_data_poly(NUM_SAMPLES, DEGREE, 0)

    # # train the model (i.e. find $\theta^*$) ... unnecessary as we have p
    # xb, yh, c = poly_fit(x_train, y_train, 
    #                   batch_size=32, epochs=10, learning_rate = 0.01, 
    #                   deg=DEGREE)

    # use gradient descent to find the x
    y_ = np.linspace(-5, 5, 20).astype(np.float32)
    c_ = p.coeffs.astype(np.float32).reshape((-1, 1))
    x_ = inverse_poly_fit(c_, y_, STEPS, LEARNING_RATE)    

    # plot results
    xb, yh = zip(*sorted(zip(x_,y_)))
    plt.plot(x_, y_, 'ro', x_train, y_train, 'b--')
    plt.show()