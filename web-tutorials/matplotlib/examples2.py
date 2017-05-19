import numpy as np
import matplotlib.pyplot as plt

#Figures usage
#Figure是GUI中的一个窗口，应用中可以创建过个Figure，他们以"figure +　ｎｕｍ"的形式命名，其中ｎｕｍ从１开始
#Figure的属性包括：序列号、大小、dpi、背景色、边缘色等，都可以通过实例化的时候传入这里参数
#Figure的关闭可以手动关闭窗口，也可以在程序中调用 close（）函数来完成，当无参数时，关闭当前figure，当参数是 num时，关闭指定窗口
#当参数是 all 时，关闭所有窗口

#Subplots usage
#子图，可以实现多个 plot 在一个窗口中以网格状呈现，使用时需要指定窗口的行列数，以及当前子图处于第几个位置。
#如 subplot（2,1,1），表示网格有两行一列，当前子图在第一个位置。第三个参数是从网格的左到右，从上到下数。


#Axes usage
#和 子图的概念很像，但是Axes可以将 plots 放与窗口中的任意位置，而subplots必须成网格状排列。因此，加入我们想要将一个小的plot放到一个大的plot中，就可以使用axes。
