#random_shuffle(
#   value,
#   seed = None,
#   name = None
#)
# 将 value 在第一维上进行随机打乱

import tensorflow as tf

a = tf.constant([[1,2,3], [4,5,6], [7,8,9]])

b = tf.random_shuffle(a)

with tf.Session() as sess:
    print(sess.run(b))
