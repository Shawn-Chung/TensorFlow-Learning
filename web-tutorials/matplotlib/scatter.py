# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

n = 1024
x = np.random.normal(0,1,n)
y = np.random.normal(0,1,n)
t = np.arctan2(y,x)

plt.axes([0.05, 0.05, 0.9, 0.9])
#alpha 是透明度
plt.scatter(x, y, s=75, c=t, alpha=0.7)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
#隐藏坐标
#plt.xticks([])
#plt.yticks([])

plt.show()