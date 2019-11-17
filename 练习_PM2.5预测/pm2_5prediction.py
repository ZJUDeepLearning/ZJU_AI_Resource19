import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

'''
李宏毅作业，pm2.5的预测，使用简单的全连接网络
考虑因素：pm2.5, SO2, N0
'''

# 训练集制作
x_input = np.zeros((480, 27))
y_input = np.zeros((480, 1))
with open('train.csv')as f:
    f_csv = csv.reader(f)
    i = 0
    for row in f_csv:
        if row[1] == 'PM2.5':
            for j in range(9):
                x_input[i*2][j] = eval(row[2+j])
                x_input[i*2+1][j] = eval(row[12+j])
            y_input[i*2][0] = eval(row[11])
            y_input[i*2+1][0] = eval(row[21])
            j = 0
        if row[1] == 'SO2':
            for j in range(9):
                x_input[i*2][j+9] = eval(row[2+j])
                x_input[i*2+1][j+9] = eval(row[12+j])
            j = 0
        if row[1] == 'NO':
            for j in range(9):
                x_input[i * 2][j + 18] = eval(row[2 + j])
                x_input[i * 2 + 1][j + 18] = eval(row[12 + j])
            j = 0

        if row[1] == 'WS_HR':
            i += 1
print('data_set mission complete')


# 神经网络设计
def add_layer(inputs, in_size, out_size, activitation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size])+0.01)
    bias = tf.Variable(tf.zeros([1, out_size])+0.1)
    wx_plus_b = tf.matmul(inputs, weights) + bias
    if activitation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activitation_function(wx_plus_b)
    return outputs


learning_rate1 = 0.01
learning_rate2 = 0.001
xs = tf.placeholder(tf.float32, [None, 27])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 27, 16, activitation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 16, 8, activitation_function=tf.nn.relu)
output = add_layer(l2, 8, 1, activitation_function=tf.nn.relu)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output), reduction_indices=[1]))
trainer1 = tf.train.AdamOptimizer(learning_rate1).minimize(loss)
trainer2 = tf.train.AdamOptimizer(learning_rate2).minimize(loss)
init = tf.initialize_all_variables()
# 神经网络设计完毕

# 开始训练
# GPU使用份额
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# loss保存列表
loss_list = []
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for t in range(20000):
        if t < 10000:
            sess.run(trainer1, feed_dict={xs: x_input, ys: y_input})
        else:
            sess.run(trainer2, feed_dict={xs: x_input, ys: y_input})
        if t % 50 == 0:
            loss_t = sess.run(loss, feed_dict={xs: x_input, ys: y_input})
            loss_list.append(loss_t)
            print('t = ' + str(t) + '   loss =' + str(loss_t))

# 画出loss图像
x_axis = np.arange(0, 20000, 50)
plt.plot(x_axis, loss_list)
plt.show()
