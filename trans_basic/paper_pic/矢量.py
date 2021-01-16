import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *
from base_trans import *
from point_project import *
from n_pt_plane import *

from o3d_impl import *

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


# 显示点云和坐标轴
def show_coord(cen_pt, vici_pts, coord, title):
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    vtx_normal = coord[:, 2]  # z_axis
    x_axis = coord[:, 0]
    y_axis = coord[:, 1]

    vici_pts = vici_pts.T
    ax = plt.figure(1).gca(projection='3d')
    # 点云
    ax.scatter(vici_pts[0], vici_pts[1], vici_pts[2], 'g', color='green', label='neighbor')  # 近邻点
    ax.scatter(cen_pt[0], cen_pt[1], cen_pt[2], 'o', color='red', label='vertex')  # 中心点

    # 坐标轴
    # x  分开画 为了得到不同的颜色
    ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],  # 起点
              # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
              x_axis[0], x_axis[1], x_axis[2],  # 对应的指向
              length=1, normalize=True, color='red')
    # y
    ax.quiver(cen_pt.T[0], cen_pt[1], cen_pt[2],  # 起点
              # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
              y_axis[0], y_axis[1], y_axis[2],  # 对应的指向
              length=1, normalize=True, color='green')
    # z
    ax.quiver(cen_pt[0], cen_pt[1], cen_pt[2],  # 起点
              # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
              vtx_normal[0], vtx_normal[1], vtx_normal[2],  # 对应的指向
              length=1, normalize=True, color='blue')
    # # z  all in one
    # ax.quiver(cen_pts.T[0], cen_pts.T[1], cen_pts.T[2],   # 起点
    #           # vtx_normal[0], vtx_normal[1], vtx_normal[2],   # 对应的指向
    #           coord[0], coord[1], coord[2],   # 对应的指向
    #           length=2, normalize=True, color='blue')

    plt.title(str(title), font1)
    ax.set_xlabel("X Axis", font1)
    ax.set_ylabel("Y Axis", font1)
    ax.set_zlabel("Z Axis", font1)

    # plt.axis("equal")
    # 设置等轴
    # s = array([-50, -25, 0, 25, 50]) * 0.5
    # scale_x = s - 30
    # scale_y = s + 25 + 60
    # scale_z = s - 25
    # plt.xticks(scale_x)
    # plt.yticks(scale_y)
    # ax.set_zticks(scale_z)
    plt.legend()
    plt.show()

    return 0


# 数据设置
# x = [0, 1, 2]
# y = x
# z = x
# 近邻点
vici_pts = loadtxt('../save_file/vic_pts_pic.txt')
# print(vici_pts)
x, y, z = vici_pts.T[0], vici_pts.T[1], vici_pts.T[2]

# 顶点
cen_pt = loadtxt('../save_file/now_pts_pic.txt')
# print(cen_pt)
u, v, w = cen_pt[0], cen_pt[1], cen_pt[2]

all_pts = vstack((cen_pt, vici_pts))

# 绘制点
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
# ax.set_aspect('equal')
# ax.set_zlim(-10, 10)
plt.title('Point Cloud Patch', font1)
ax.legend()  # Add a legend.
plt.show()


# 找到拟合的坐标系
coord = get_coord(cen_pt, vici_pts)
vtx_normal = coord[:, 2]
x_axis = coord[:, 0]
y_axis = coord[:, 1]
# print(norm(vtx_normal))
# print(coord)

# print('vtx:', cen_pt)

show_coord(cen_pt, vici_pts, coord, title='Tangent Plane Estimation')

# 将点投影到平面
# 找到平面
plan = get_plan(vtx_normal, cen_pt)
print('plan:', plan)

# 投影到平面
pts_buff = vstack((cen_pt, vici_pts))  # 所有近邻点
# pts_buff = vici_pts
# print('pts_buff_投影前:\n', pts_buff)
pts_buff = pt_to_plane(pts_buff, plan, vtx_normal)  # 这里 投影点改成矩阵输入形式
# print('pts_buff_投影后:\n', pts_buff)

# 可视化投影后的点
show_coord(pts_buff[0], pts_buff[1:], coord, 'Projected Point Cloud Patch')

# 平面反变换 将平面旋转到XOY
coord_inv = inv(coord)
coord_inv_vis = dot(coord_inv, coord)
roto_pts = dot(coord_inv, pts_buff.T).T  # 将平面旋转到与z平行
pts_buff = roto_pts

show_coord(roto_pts[0], roto_pts[1:], coord_inv_vis, 'Oriented Tangent Plane')

# pts_buff[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动
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
plt.title('Delaunay Triangulate')
plt.show()

# 通过三角形索引构建mesh

# 根据顶点和三角形索引创建mesh
mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

# 求mesh normal
mesh.compute_triangle_normals()
mesh_normals = array(mesh.triangle_normals)
# print(mesh_normals)

# mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

# 可视化mesh
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

# 顶点
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(all_pts)
array(pcd2.colors)[:1] = [0, 0, 1]
array(pcd2.colors)[1:] = [0, 1, 0]
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=5))

o3d.visualization.draw_geometries([mesh,
                                   pcd2,
                                   # axis_pcd,
                                   # mesh1,
                                   # mesh2
                                   ],
                                  window_name='ANTenna3D',
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 2.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]
                                  # point_show_normal=True
                                  )

