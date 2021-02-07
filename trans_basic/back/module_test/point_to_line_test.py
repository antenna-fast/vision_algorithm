from numpy import *
import matplotlib.pyplot as plt

from point_project import *

l = array([1, 1, 1])
l_n = array([l[0], l[1]])

# 生成数据点
l_x = array(range(-10, 10))
l_y = -1 * (l[0] * l_x + l[2]) / l[1]

# 投影点
p = array([-5, -5])


x = pt_to_line(p, l)

# 看上去不是正交投影，坐标轴长度需要统一
plt.scatter(l_x, l_y)
plt.scatter(p[0], p[1])
plt.scatter(x[0], x[1])

# plt.xticks(fontproperties='Times New Roman', size=10)  # 设置坐标轴刻度文字的尺寸
# plt.yticks(fontproperties='Times New Roman', size=10)  # 设置坐标轴刻度文字的尺寸
plt.axis("equal")
plt.show()

# 将邻域中的多个点投影到平面

# 空间平面上的点的老内

# 根据三角化索引找到三角形索引
