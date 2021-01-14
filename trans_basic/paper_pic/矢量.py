import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *
from base_trans import *
from point_project import *
from n_pt_plan import *


# fig = plt.figure()
# ax = fig.gca(projection='3d')

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
# 近邻点
vici_pts = loadtxt('../save_file/vic_pts_pic.txt').T
# print(vici_pts)
x, y, z = vici_pts[0], vici_pts[1], vici_pts[2]

# 顶点
cen_pt = loadtxt('../save_file/now_pts_pic.txt').T
# print(cen_pt)
u, v, w = cen_pt[0], cen_pt[1], cen_pt[2]

# 绘制点
# plt.scatter(x, y, z)

# ax.quiver(x, y, z,   # 起点
#           u, v, w,   # 对应的指向
#           length=2, normalize=True)


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }

ax = plt.figure(1).gca(projection='3d')

ax.plot(x, y, z, 'g.', color='green',  label='neighbor')  # 近邻点
ax.plot(u, v, w, 'o', color='red', label='vertex')  # 中心点
# ax.plot(x[0], x[1], x[2], 'r.')
ax.set_xlabel("X Axis", font1)
ax.set_ylabel("Y Axis", font1)
ax.set_zlabel("Z Axis", font1)

# plt.axis("equal")
# plt.axis("auto")
# ax.set_aspect('equal')
# ax.set_zlim(-10, 10)
plt.title('Point Cloud Patch', font1)
ax.legend()  # Add a legend.
plt.show()


# 找到拟合的坐标系
coord = get_coord(cen_pt.T, vici_pts.T)
vtx_normal = coord[:, 2]
print(norm(vtx_normal))
print(coord)

pts_buff = vstack((cen_pt.T, vici_pts.T))

coord_inv = inv(coord)  # 反变换
roto_pts = dot(coord_inv, pts_buff.T).T  # 将平面旋转到与z平行
pts_buff = roto_pts

pts_buff[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动

pts_2d = pts_buff[:, 0:2]

# 三角化 找到拓扑结构
tri = Delaunay(pts_2d)
# print(dir(tri))
# print(tri.points)
# print(len(tri.points))
tri_idx = tri.simplices
# print(tri_idx)  # 三角形索引

plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
plt.show()

# 找到平面
plan = get_plan(vtx_normal, vici_pts)
