#利用cnn实现数字识别



# -*-coding: utf-8 -*-

import tensorflow as tf
import input_data

#import image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#使用InteractiveSession可以在计算图的过程中插入计算图，而使用Session则需要在启动之前构建整个计算图
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])


#定义函数来初始化变量(正态分布)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#定义卷积和池化函数,使用1步长、0边距的模板
#conv2d的输入 x和W必须是 4-D tensor
#x = [batch, in_height, in_width, in_channels]
#W = [filter_height, filter_width, in_channels, out_channels]
#strides = [1, stride, dtride, 1] 表示卷积核的水平和垂直步长
#padding = 'SAME' 或者 'VALID' 表示边缘填补方式
#返回和x一样type的tensor
def conv2d(x, W):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#x 的 shape是[batch, in_height, in_width, in_channels],type is float32
#ksize is a list of ints whos size >=4.代表输入tensor的每一维的窗口大小
#strides is a list of ints whos size >=4.代表输入tensor在每一维的窗口滑动步长大小
#max_pool 及对输入tensor 进行取2x2窗口中的最大值，将输入tensor缩小为原来的 1/4
def max_pool_2x2(x):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#第一层卷积，由一个卷积加一个池化完成
#卷积模板为5x5，输入为一个通道（也就是一张单通道图片），输出是32个通道（也就是第一层有32个神经元）
with tf.name_scope('W_conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    tf.summary.histogram('W_conv1' + 'Weights', W_conv1)
#所以有32个偏移变量
with tf.name_scope('b_conv1'):
    b_conv1 = bias_variable([32])
    tf.summary.histogram('b_conv1' + 'bias', b_conv1)

#改变输入图片的形状，第一维无用，第二、三维是图片的长宽，第四维是图片的通道数（如果是RGB图像，则为3）
x_image = tf.reshape(x, [-1, 28, 28, 1])

#进行卷积操作并用relu激活，最后使用池化操作
with tf.name_scope('relu'):
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 为 [-1, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
#输出为64通道，也就是64个神经元
with tf.name_scope('W_conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    tf.summary.histogram('W_conv2' + 'Weights', W_conv2)
with tf.name_scope('b_conv2'):
    b_conv2 = bias_variable([64])
    tf.summary.histogram('b_conv2' + 'bias', b_conv2)
with tf.name_scope('relu'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第三层 全连接层,到现在，经过两次2x2的池化，图片由28x28变成了7x7大小
#这一层使用了1024个神经元来实现全连接
#这一层的输入总共有 7 * 7 * 64 个像素点
with tf.name_scope('W_fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    tf.summary.histogram('W_fc1' + 'W', W_fc1)
with tf.name_scope('b_fc1'):
    b_fc1 = bias_variable([1024])
    tf.summary.histogram('b_fc1' + 'b', b_fc1)
#对这一层的输入 reshape，原来是[-1, 7, 7, 64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
with tf.name_scope('relu'):
    with tf.name_scope('xW_plus_b'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，在输出层之前加入dropout，用一个placeholder代表一个神经元的输出在dropout中保存不变的概率。
#这样就可以在训练的时候启用dropout，在测试的时候关闭它。
#TensorFlow中的dropout除了可以屏蔽神经元的输出外，还会处理输出值的scale，所以使用dropout的时候就不考虑scale了。
keep_prob = tf.placeholder("float")
with tf.name_scope('dropout'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层 为 softmax
with tf.name_scope('W_fc2'):
    W_fc2 = weight_variable([1024, 10])
    tf.summary.histogram('W_fc2' + 'W', W_fc2)
with tf.name_scope('b_fc2'):
    b_fc2 = bias_variable([10])
    tf.summary.histogram('b_fc2' + 'b', b_fc2)

with tf.name_scope('softmax'):
    with tf.name_scope('xW_plus_b'):
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#开始训练
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prodiction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prodiction, "float"))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
        print ("step %d, trainning accuracy %g" %(i, train_accuracy))
        result = sess.run(merged, feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
        writer.add_summary(result, i)        
    train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})
    
print("test accuracy %g" %accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
