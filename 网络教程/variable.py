#variable 变量，是TensorFlow中的一个类，定义一个变量时就相当于构建了一个类实例
#调用构造函数Variable()，该函数需要一个初始化的值，该值可以是任何数据类型和形状
#变量定义后，其数据类型和形状就固定了，但是其值可以改变，使用assign()函数

import tensorflow as tf

#create a variable 2*2
w = tf.Variable([[1],[2]], name='shawn', dtype = tf.float32)
b = tf.Variable([[2,4]], dtype = tf.float32)
print(w)
print(b)
#矩阵乘法
y = tf.matmul(w, b)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(y))
x = w.assign_add([[2],[3]])
print(sess.run(x))
