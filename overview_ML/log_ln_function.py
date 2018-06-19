import math
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__': #画出log函数
    #设置x 和 y的值
    x = [float(i)/100.0 for i in range(1,300)]
    y = [math.log(i) for i in x]
    plt.plot(x, y , 'r-', linewidth=3, label='log Curve')
    a = [x[20], x[175]]
    b = [y[20], y[175]]
    plt.plot(a, b, 'g-', linewidth=2)
    plt.plot(a, b, 'b*', markersize=15, alpha=0.75)
    #设置线条说明
    plt.legend(loc='upper left')
    plt.grid(True)#是否有网格
    plt.xlabel('x')
    plt.ylabel('log(x)')
    plt.show()

    #画sigmoid函数
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-h))
    h = np.arange(-10, 10, 0.1) # 定义x的范围，像素为0.1
    s_h = sigmoid(h) # sigmoid为上面定义的函数
    plt.plot(h, s_h)
    plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴--这一句没有作用（两条线没有画出来）
    plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
    plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
    plt.ylim(-0.1, 1.1) # 加y轴范围
    plt.xlabel('h')
    plt.ylabel('$S(h)$')
    plt.show()