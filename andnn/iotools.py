import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
# %matplotlib inline


def cubic_fit(x_train, y_train, batch_size, epochs):
    # pick randomize the initial guess and set up inputs
    c = tf.Variable(tf.truncated_normal([4, 1]))
    x = tf.placeholder(tf.float32, (batch_size,))
    y = tf.placeholder(tf.float32, (batch_size,))
    X = tf.stack([tf.ones(tf.shape(x)), x, x**2, x**3], axis=1)
    
    # compute predictions and loss
    y_hat = tf.squeeze(tf.matmul(X, c))  # `tf.squeeze` is similar to `np.ravel`
    loss = tf.reduce_sum((y - y_hat)**2)

    # To train: specify an optimization scheme and add it to the `fetches`
    # all "trainable" variables will be trained (here that's just `c`)
    update_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    
    batch_size = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize weights (i.e. initial guess for c)
        for epoch in range(epochs):
            for step in range(num_samples//batch_size):
                x_batch = x_train[step*batch_size: (step + 1)*batch_size]
                y_batch = y_train[step*batch_size: (step + 1)*batch_size]

                y_hat_, loss_, _ = sess.run(fetches=[y_hat, loss, update_step], 
                                                    feed_dict={x: x_batch, y: y_batch})
                update_plots(x_batch, y_batch, y_hat_, loss_)
            
            # shuffle
            permutation = np.random.permutation(len(x_train))
            x_train, y_train = x_train[permutation], y_train[permutation]
            print("epoch: %s | loss/sample = %s" %(epoch, loss_/batch_size))


# let's make some fake data that's could reasonably be approximated by a cubic
npr = np.random
num_samples = 10000
cubic = np.poly1d(1 + npr.randint(-10, 10, (3,)))      # a random cubic
x_train = npr.rand(num_samples)                        # training data
y_train = (cubic(x_train) + npr.rand(num_samples)/10)  # training ground truth


# some matplotlib junk 
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
gt_curve,   = ax1.plot(x_train, y_train, 'bo')
pred_curve, = ax1.plot(x_train, y_train , 'r*')
loss_curve, = ax2.plot([], 'g-')


def update_plots(x_batch, y_batch, y_hat_, loss_):
    pred_curve.set_xdata(x_batch)
    pred_curve.set_ydata(y_hat_)
    loss_curve.set_ydata(loss_)
    fig.canvas.draw()
    fig.canvas.flush_events()


cubic_fit(x_train, y_train, batch_size=100, epochs=100)
