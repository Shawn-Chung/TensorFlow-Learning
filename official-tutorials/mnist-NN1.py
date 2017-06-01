#该程序 使用 tensorflow中的 Nearest Neighbor Classifier （最近邻分类器）来对字符的识别
#accuracy: 0.9616

# -*-coding: utf-8 -*-

import input_data
import numpy as np
import tensorflow as tf

#import image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(55000)
Xte, Yte = mnist.test.next_batch(10000)

# 计算图输入占位符
xtrain = tf.placeholder("float", [None, 784])
xtest = tf.placeholder("float", [784])

# 使用L1距离进行最近邻计算
# 计算L1距离
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)

# 预测: 获得最小距离的索引 (根据最近邻的类标签进行判断)
pred = tf.arg_min(distance, 0)

# 初始化节点
init = tf.global_variables_initializer()

#最近邻分类器的准确率
accuracy = 0.

# 启动会话
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xte)  #测试样本的数量
    # 在测试集上进行循环
    for i in range(Ntest):
        # 获取当前测试样本的最近邻
        nn_index = sess.run(pred, feed_dict={xtrain: Xtr, xtest: Xte[i, :]})
        # 获得最近邻预测标签，然后与真实的类标签比较
        pred_class_label = np.argmax(Ytr[nn_index])
        true_class_label = np.argmax(Yte[i])
        print("Test", i, "Predicted Class Label:", pred_class_label,
              "True Class Label:", true_class_label)
        # 计算准确率
        if pred_class_label == true_class_label:
            accuracy += 1
    print("Done!")
    accuracy /= Ntest
    print("Accuracy:", accuracy)