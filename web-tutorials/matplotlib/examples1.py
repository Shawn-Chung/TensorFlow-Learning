#matplotlib使用教程1：一维函数的绘制和一些基本操作的演示。
#设置坐标显示范围
#设置坐标显示的点
#移动坐标轴
#为图添加标签
#为部分点添加标注

import numpy as np
import matplotlib.pyplot as plt

#在默认的figure上画图
x = np.linspace(-np.pi, np.pi, 50)
y = 3*x + 2

plt.plot(x, y, color = "blue", linewidth = 2.0, linestyle = "-", label='f(x)=3x+2')

#设置x、y坐标的显示范围
#plt.xlim(-1, 1)
#plt.ylim(-3, 3)

#设置x、y坐标上需要显示的坐标点,在 -2到2的范围里显示11个点
#plt.xticks(np.linspace(-2,2,11, endpoint=True))
#plt.xticks([-2, -1.1, 0, 1.4, 2])
#设置坐标显示的点时同时设置显示的内容，注意这里的内容格式
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

#移动坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

#在图中添加说明,需要再调用plot函数的时候传入 label 参数
plt.legend(loc='upper left', frameon = False)

#在图中标注部分点
t = np.pi/3
plt.plot([t,t], [0,3*t+2], linewidth=1.5, linestyle='-.', color='blue')
plt.scatter([t,], [3*t+2,], 50, color='blue')
plt.annotate(r'$f(\frac{\pi}{3})=\pi+2$', #要显示的内容
             xy=(t, 3*t+2), 
             xycoords='data', 
             xytext=(+40,-20), #文本相对于改点的坐标
             textcoords='offset points', 
             fontsize=18, #文本的字体大小
             arrowprops = dict(arrowstyle="->", connectionstyle="arc3, rad=0.5"))

plt.show()