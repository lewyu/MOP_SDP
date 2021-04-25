import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def Read_Files(filename):
    X_axis = []  # X
    Y_axis = []  # Y
    Z_axis = []  # Z
    with open(filename, 'r') as f:
        for line in f.readlines():
            # print(line)
            x = line.split("\t")[0]  # 注意，这里不是使用空格，而是使用Tab制表符进行分割
            # print(x)
            y = line.split("\t")[1]
            # print(y)
            z = line.split("\t")[2]
            # print(z)
            X_axis.append(float(x))
            Y_axis.append(float(y))
            Z_axis.append(float(z))
    f.close()
    return X_axis, Y_axis, Z_axis


def plot_PF(X_axis, Y_axis, Z_axis):
    ax = plt.figure().add_subplot(111, projection='3d')
    # c 设置颜色，alpha设置透明度，s设置点的大小
    ax.scatter(X_axis, Y_axis, Z_axis, c='b', alpha=0.5, s=3)

    plt.savefig(Figname + '.png', dpi=600)
    plt.show()


Filename = './data/hType_sphere.pf'
Figname = 'hType_sphere'
X_axis, Y_axis, Z_axis = Read_Files(Filename)
plot_PF(X_axis, Y_axis, Z_axis)
