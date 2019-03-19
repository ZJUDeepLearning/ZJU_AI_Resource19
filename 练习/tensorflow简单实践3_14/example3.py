# 例子三
# 利用TensorFlow训练神经网络实现手写数字识别
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# this is data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# draw loss and accuracy
loss_list = []
accuracy_list = []


def add_layer(inputs, in_size, out_size, activitation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size])+0.1)
    wx_plus_b = tf.matmul(inputs, weights) + bias
    if activitation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activitation_function(wx_plus_b)
    return outputs


# structure
xs = tf.placeholder(tf.float32, [None, 784])    # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
l1 = add_layer(xs, 784, 128, activitation_function=tf.nn.tanh)
prediction = add_layer(l1, 128, 10, activitation_function=tf.nn.softmax)

# cross entropy
# loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices=[1]))
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    for i in range(10000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            accuracy_ = sess.run(accuracy, feed_dict={xs: data.test.images, ys: data.test.labels})
            print(accuracy_)
            accuracy_list.append(accuracy_)
            loss_list.append(sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys}))

x = np.arange(0, 10000, 50)
plt.subplot(211)
plt.plot(x, loss_list)
plt.subplot(212)
plt.plot(x, accuracy_list)
plt.show()

print('the procession is done')
