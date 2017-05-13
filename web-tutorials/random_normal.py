# -*- coding: utf-8 -*-
#random_normal(   产生随机正太分布数据
#   shape,
#   mean = 0.0,  平均值
#   stddev = 1.0,  标准差
#   dtype = tf.float32,
#   seed = None,  种子
#   name = None
#)
# 使用平均值和标准差来控制输出值得范围

import tensorflow as tf

a = tf.random_normal(shape = (3,3), mean = 5.1, stddev = 9)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
