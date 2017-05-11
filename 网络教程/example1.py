#using coding:utf-8

import tensorflow as tf
import numpy as np

#cerate data  the shape is 1 * 100, type is float32
#the np.random.rand product data in [0,1)
x_data = np.random.rand(100).astype(np.float32)
print (np.version.version)
y_data = x_data * 0.5 + 3.2
#定义变量
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化所有变量，前面定义了变量，这里就一定要初始化
init = tf.global_variables_initializer()

sess = tf.Session()
#前面有定义变量，这里也一定要run一下
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(weights), sess.run(biases))
