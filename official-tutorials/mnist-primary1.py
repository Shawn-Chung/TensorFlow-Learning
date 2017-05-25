#该程序同 mnist-primary.py原理相同，只是使用了InteractiveSession会话

# -*-coding: utf-8 -*-

import tensorflow as tf
import input_data

#import image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#使用InteractiveSession可以在计算图的过程中插入计算图，而使用Session则需要在启动之前构建整个计算图
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict = {x: batch[0], y_:batch[1]})
    
    if i%50 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))    