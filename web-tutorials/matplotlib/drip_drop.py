#用花点的方式生成雨点动图，采用了animation模块的FuncAnimation函数
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#create a blank figure  with white background
fig = plt.figure(figsize=(6,6), facecolor='white')

#create a new axes over the whole figure, no frame and a 1:1 aspect ratio
#第一个参数是当前axes在 fig中的位置[x,y,w,h]，第二个参数是axes的边框是否显示
ax = fig.add_axes([0.2,0.1,0.75,0.75], frameon=True, aspect=1)

#create some rings
n = 50
size_min = 50
size_max = 50*50

#ring position
P = np.random.uniform(0,1,(n,2))
#ring colors
C = np.ones((n,4)) * (0,0,0,1)
#alpha color channel goes from 0 to 1
C[:,3] = np.linspace(0,1,n)
#ring sizes
S = np.linspace(size_min, size_max, n)
#scatter plot
scat = ax.scatter(P[:,0], P[:,1], s= S, lw=0.5, edgecolors = C, facecolors='None')
#ensure limits are [0,1] and remove ticks
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([])
ax.set_yticks([])



#define the update function 
def update(frame):
    global P,C,S
    #every ring is made more transparent 制作颜色渐变的效果，从亮到模糊
    C[:,3] = np.maximum(0, C[:,3] - 1.0/n)
    #each ring is made larger  制作size渐变的效果，从小到大
    S += (size_max - size_min) / n
    
    #reset ring specific ring,
    #因为第一帧时（初始化ring时）从第一个到第50个ring的颜色是逐渐变亮的，因此第一个ring会首先消失，后面依次
    #下面四句每一帧处理一个点，这个点正好是当前消失的点，这里重新生成position，size设为最小，color设为最亮
    i = frame % 50
    P[i] = np.random.uniform(0,1,2)
    S[i] = size_min
    C[i,3] = 1
    
    #update scatter object
    scat.set_edgecolors(C)
    scat.set_sizes(S)
    scat.set_offsets(P)
    
    #return the modified object
    #return 一定要加逗号，否则FuncAnimation不能正常运行？？
    return scat,


animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)
#save it
#animation.save('rain.gif', writer='imagemagick', fps=30, dpi=40)

plt.show()

