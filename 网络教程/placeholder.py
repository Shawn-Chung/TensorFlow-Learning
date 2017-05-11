#placeholder(
#   dtype,
#   shape  = None,
#   name = None
#)
#定义一个占位符tensor，具体值需要再运算时使用 feed_dict 作为 Session.run()的参数指定

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape = (3, 3))
y = tf.matmul(x, x)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
#    print(sess.run(y)) #error,because x was not fed.

    rand_array = np.random.rand(3,3)
    print(sess.run(y, feed_dict = {x: rand_array}))
