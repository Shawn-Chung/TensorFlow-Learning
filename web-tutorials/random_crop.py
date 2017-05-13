# -*- coding: utf-8 -*-
#random_crop(
#   value,
#   size,
#   seed = None,
#   name = None
#)
#随机从 value 中裁剪出 size 大小的 tensor

import tensorflow as tf

a = tf.constant([[1,2,3,4,5],
                 [6,7,8,9,10]])

b = tf.random_crop(a, [2,2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
