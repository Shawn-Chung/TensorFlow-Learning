# -*-coding: utf-8 -*-

import tensorflow as tf
import input_data

#import image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#create a tensor to hold input data(just a shape, has no values)
x = tf.placeholder("float", [None, 784], name = 'input')

#create tensor W and b, and init to all 0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#realize the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#define the loss function with cross-entropy
y_ = tf.placeholder("float", [None, 10])

#定义代价函数：交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#define the train method
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#init all the variables
init = tf.initialize_all_variables()

#start
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_:batch_ys})

    if i%50 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
