# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

n = 12
x = np.arange(n)
y1 = (1-x/float(n)) * np.random.uniform(0.5, 1.0, n)
y2 = (1-x/float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(x, +y1, facecolor='#9999ff', edgecolor='blue')
plt.bar(x, -y2, facecolor='#ff9999', edgecolor='white')

for a,b in zip(x,y1):
    plt.text(a+0.04, b+0.05, '%.2f' % b, ha='center', va='bottom')
for a,b in zip(x,y2):
    plt.text(a+0.04, -b-0.05, '%.2f' % b, ha='center', va='top')

plt.ylim(-1.25, 1.25)

plt.show()