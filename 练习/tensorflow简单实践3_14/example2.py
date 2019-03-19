# 例子二
# 利用TensorFlow训练神经网络模拟二/三次函数（线性回归）

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activitation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])+0.1)
    wx_plus_b = tf.matmul(inputs, weights) + bias
    if activitation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activitation_function(wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) * x_data - 0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activitation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activitation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    for i in range(10000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
            '''
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            '''
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=4)
            plt.pause(0.1)
            ax.lines.remove(lines[0])
    plt.show()
