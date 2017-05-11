#constant 常量，定义时constant()构造函数，需要传入初始值和shape，其中shape是可选的
#如果没有指定shape，则使用初始值的shape，如果有指定shape，则初始值的shape不能大于指定
#的shape，当初始值的shape小于指定的shape时，则用初始值的最后一个数填充


import tensorflow as tf
#output is [1,2,3,4,5]
a = tf.constant([1,2,3,4,5])
print(a)
#output is [[1,2,3,4,5], [5,5,5,5,5], [5,5,5,5,5], [5,5,5,5,5], [5,5,5,5,5]]
b = tf.constant([1,2,3,4,5], shape = [5,5])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
