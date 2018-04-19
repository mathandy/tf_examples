import tensorflow as tf
import matplotlib.pyplot as plt
# plt.ion()
import numpy as np


NUM_SAMPLES = 2**13  # number of fake training examples
DEGREE = 3  # degree of polynomial 
NOISE_FACTOR = 10  # how much noise in training examples 


def poly_fit(x_train, y_train, batch_size, epochs, learning_rate, deg=3):
    """Fits a polynomial to the input data using stochastic gradient 
    descent."""

    # pick random initial guess for `c` and set up inputs
    c = tf.Variable(tf.truncated_normal([deg + 1, 1]))
    x = tf.placeholder(tf.float32, (batch_size,))
    y = tf.placeholder(tf.float32, (batch_size,))

    # transform this into a linear regression problem
    X = [tf.ones(tf.shape(x)), x] + [x**k for k in range(2, deg+1)]
    X = tf.stack(X, axis=1)
    
    # compute predictions and loss
    y_hat = tf.squeeze(tf.matmul(X, c))  # `tf.squeeze` is similar to `np.ravel`
    loss = tf.reduce_sum((y - y_hat)**2)/batch_size

    # To train: specify an optimization scheme and add it to the `fetches`
    # all "trainable" variables will be trained (here that's just `c`)
    update_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize weights (i.e. guess for c)
        for epoch in range(epochs):
            # shuffle
            permutation = np.random.permutation(len(x_train))
            x_train, y_train = x_train[permutation], y_train[permutation]
            
            # train for an epoch
            for step in range(len(x_train)//batch_size):
                x_batch = x_train[step*batch_size: (step + 1)*batch_size]
                y_batch = y_train[step*batch_size: (step + 1)*batch_size]

                y_hat_, loss_, _ = sess.run(fetches=[y_hat, loss, update_step], 
                                                    feed_dict={x: x_batch, y: y_batch})
            
            print("epoch: %s | loss = %s" %(epoch, loss_))

        parameters = sess.run(c)
    return x_batch, y_hat_, parameters
        

# let's make some fake data that's could reasonably be approximated by a cubic
def gen_fake_data_poly(num_samples=2**13, deg=3, noise_factor=10):
    npr = np.random
    p = np.poly1d(1 + npr.randint(-10, 10, (deg+1,)))
    x_train = np.linspace(np.roots(p).real.min() - 1, 
                          np.roots(p).real.max() + 1, num_samples)
    y_train = (p(x_train) + npr.rand(num_samples)/noise_factor)
    return x_train, y_train, p

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