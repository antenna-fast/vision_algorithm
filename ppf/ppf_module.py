# ppf　特征计算模块

import numpy as np
from numpy import *


# 计算连接两点向量
pt1 = np.array([0, 1])
pt2 = np.array([1, 8])
link_vector = pt1 - pt2
# print('连线：', link_vector)

# 计算夹角
# 原理：
#   向量内积的几何意义：向量a在b上的投影
a = np.array([1, 0, 0])
b = np.array([1, 1, 0])
# 弧度制
theta = np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print('弧度制：', theta)
print('角度：', theta * 180 / np.pi)

# 外积  可以将a写成反对陈矩阵，然后写成矩阵与向量b的乘法
# 几何意义：
#   两个向量张成四边形的有向面积，只对三维向量有意义
# c=axb
c = np.cross(a, b)
print(c)


# 通过反正切计算 直接算出来了每个轴上面的夹角
angle = arctan(cross(a, b) / dot(a, b))
angle = 180 / pi * angle
print('ang:', angle)

# 所有的点对：还是判断一下 i != j
