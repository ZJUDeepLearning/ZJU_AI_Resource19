# 例子一
# 利用TensorFlow模拟简单的一次函数

import numpy as np
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = 3 * x_data + 1.5

# create structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Bias = tf.Variable(tf.zeros([1]))

y_predict = Weights * x_data + Bias
loss = tf.reduce_mean(tf.square(y_predict-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init_variables = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_variables)

for step in range(400):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Bias))
