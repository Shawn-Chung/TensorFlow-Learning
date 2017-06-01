#该程序 使用Nearest Neighbor Classifier （最近邻分类器）来对字符的识别
#accuracy: 0.992320

# -*-coding: utf-8 -*-

import input_data
import numpy as np

#import image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch = mnist.train.next_batch(55000)
Xtr = batch[0]
Ytr = batch[1]

batch_test = mnist.test.next_batch(10000)
Xte = batch_test[0]
Yte = batch_test[1]

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        #待预测的样本数量
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros([num_test, 10])

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            #计算每个测试样本同训练数据集中的每个样本之间的L1距离
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            #找到距离最小的一个训练样本
            min_index = np.argmin(distances) # get the index with smallest distance
            #该样本所属类别就是预测类别
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            print("%i testing..."%i)
            print(Ypred[i])

        return Ypred
    
    
    
nn = NearestNeighbor()
nn.train(Xtr, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte)
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))