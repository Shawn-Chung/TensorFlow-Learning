# -*- coding: utf-8 -*-
#random_uniform(
#   shape,
#   minval = 0,
#   maxval = None,
#   dtype = tf.float32,
#   seed = None,
#   name = None
#)
# 产生随机均匀分布的数据，介于 [minval , maxval) 之间,默认是 [0, 1) 之间

import tensorflow as tf

a = tf.random_uniform(shape = (3,3), minval = 5, maxval = 10)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
