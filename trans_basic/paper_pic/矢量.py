import matplotlib.pyplot as plt
from numpy import *

fig = plt.figure()
ax = fig.gca(projection='3d')

# # Make the grid
# x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.8))
#
# # Make the direction data for the arrows
# u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
# v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
#      np.sin(np.pi * z))

# 数据设置
# x = [0, 1, 2]
# y = x
# z = x
# 顶点
x, y, z = loadtxt('../save_file/now_pts_pic.txt')
# print(x, y, z)

# 近邻点
u = 1
v = 1
w = 1

# 坐标轴设置

ax.quiver(x, y, z,   # 起点
          u, v, w,   # 对应的指向
          length=0.5, normalize=True)

plt.show()
