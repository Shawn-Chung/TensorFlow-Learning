# -*- coding: utf-8 -*-
#根据样本数据训练二次函数曲线


import tensorflow as tf
import numpy as np
#引入图形库
import matplotlib.pyplot as plt

#定义添加神基层的函数
def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    layer_name = "layer%s"%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
            #记录变量的变化过程
            tf.summary.histogram(layer_name + 'Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_puls_b = tf.matmul(inputs, Weights) + biases

        if activation_func is None:
            outputs = Wx_puls_b
            tf.summary.histogram(layer_name + 'outputs', outputs)
        else:
            outputs = activation_func(Wx_puls_b)
            tf.summary.histogram(layer_name + 'outputs', outputs)

        return outputs
    
#产生均分向量
x_data = np.linspace(-2, 2, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#使用with name_scope 实现神经网络模型可视化
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')
#添加两层网络
l1 = add_layer(xs, 1, 20, n_layer = 1, activation_func = tf.nn.relu)
prediction = add_layer(l1, 20, 1, n_layer = 2, activation_func = None)
#定义代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                      reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
#定义训练的目标，使用梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
#将神经网络的模型图保存到本地，可以使用tensorboard在浏览器中打开查看
#将所有的记录合并到一起
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)
sess.run(init)

fig = plt.figure()
#添加子图
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
#设置y方向上的显示范围
plt.ylim(-1,4)
plt.show()

#begain to train
for i in range(1000):
    sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
    if i%50 == 0:
        #to see the step improvement
        print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
        result = sess.run(merged, feed_dict = {xs:x_data, ys:y_data})
        writer.add_summary(result, i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        plt.pause(0.1)
