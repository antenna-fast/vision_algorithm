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
# plt.show()


# 找到拟合的坐标系
coord = get_coord(cen_pt.T, vici_pts.T)
vtx_normal = coord[:, 2]
x_axis = coord[:, 0]
y_axis = coord[:, 1]
print(norm(vtx_normal))
print(coord)

print('vtx:', cen_pt)
ax = plt.figure(1).gca(projection='3d')

ax.scatter(x, y, z, 'g', color='green',  label='neighbor')  # 近邻点
ax.scatter(u, v, w, 'o', color='red', label='vertex')  # 中心点

# 坐标轴
cen_pts = array([cen_pt, cen_pt, cen_pt])
print('cen_pts:', cen_pts)
# x  分开画 为了得到不同的颜色
ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          x_axis[0], x_axis[1], x_axis[2],   # 对应的指向
          length=2, normalize=True, color='red')
# y
ax.quiver(cen_pt.T[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          y_axis[0], y_axis[1], y_axis[2],   # 对应的指向
          length=2, normalize=True, color='green')
# z
ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
         vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          length=2, normalize=True, color='blue')
# # z  all in one
# ax.quiver(cen_pts.T[0], cen_pts.T[1], cen_pts.T[2],   # 起点
#           # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
#           coord[0], coord[1], coord[2],   # 对应的指向
#           length=2, normalize=True, color='blue')
plt.show()


# 将点投影到平面
# 找到平面
plan = get_plan(vtx_normal, cen_pt)
print('plan:', plan)

# 投影到平面
# pts_buff = vstack((cen_pt.T, vici_pts.T))  # 所有近邻点
pts_buff = vici_pts.T
# print('pts_buff_投影前:\n', pts_buff)
pts_buff = pt_to_plan(pts_buff, plan, vtx_normal)  # 这里 投影点改成矩阵输入形式
# print('pts_buff_投影后:\n', pts_buff)

# 可视化投影后的点
ax = plt.figure(1).gca(projection='3d')
ax.scatter(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g', color='green',  label='neighbor')  # 近邻点
ax.scatter(cen_pt.T[0], cen_pt.T[1], cen_pt.T[2], 'o', color='red',  label='neighbor')  # 近邻点
# 坐标系
# x
ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          x_axis[0], x_axis[1], x_axis[2],   # 对应的指向
          length=2, normalize=True, color='red')
# y
ax.quiver(cen_pt.T[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          y_axis[0], y_axis[1], y_axis[2],   # 对应的指向
          length=2, normalize=True, color='green')
# z
ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],   # 起点
          # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
         vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
          length=2, normalize=True, color='blue')

plt.show()

coord_inv = inv(coord)  # 反变换
roto_pts = dot(coord_inv, pts_buff.T).T  # 将平面旋转到与z平行
pts_buff = roto_pts

# ax = plt.figure(1).gca(projection='3d')
# ax.scatter(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g', color='green',  label='neighbor')  # 近邻点
# plt.show()

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
plt.title('Delaunay')
# plt.show()

