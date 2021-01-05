from numpy import *
import numpy as np
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay

import open3d as o3d

import copy

# 定义平面方程
p = array([1, 1, 1, -10])  # z如果相对于点的长宽太小 看上是过原点
p_n = p[:3]
# 注意 平面要按照方程生成！
pts_buff = []
for i in range(-10, 10):
    for j in range(-10, 10):
        z = -1 * (p[3] + p[0] * i + p[1] * j) / p[2]  # 问题:如果c=0 就不行了
        pts_buff.append([i, j, z])
        # pts_buff.append([i, j, abs(z)])

pts_buff = array(pts_buff)
pts = copy.deepcopy(pts_buff)


# 求出平面坐标系
def get_coord(pts):
    U, s, Vh = svd(pts)  # U.shape, s.shape, Vh.shape
    # print('U:\n', U)
    # print('s:\n', s)
    # print('Vh:\n', Vh)
    # 排序索引 由小到大
    # x_axis, y_axis, z_axis = Vh  # 由大到小排的 对应xyz轴
    return Vh.T  # 以列向量表示三个坐标轴


def get_non_manifold_vertex_mesh(verts, triangles):
    # verts = np.array(
    #     [
    #         [-1, 0, -1],
    #         [1, 0, -1],
    #         [0, 1, -1],
    #         [0, 0, 0],
    #         [-1, 0, 1],
    #         [1, 0, 1],
    #         [0, 1, 1],
    #     ],
    #     dtype=np.float64,
    # )

    # triangles = np.array([
    #     [0, 1, 2],
    #     [0, 1, 3],
    #     [1, 2, 3],
    #     [2, 0, 3],
    #     [4, 5, 6],
    #     [4, 5, 3],
    #     [5, 6, 3],
    #     [4, 6, 3],
    # ])
    #

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    # mesh.rotate(
    #     mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)),
    #     center=mesh.get_center(),
    # )
    return mesh


coord = get_coord(pts_buff)  # 平面的旋转坐标变换

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


# plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
# plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
# plt.show()


# 此处的输入点应当是原始点云中的

mesh = get_non_manifold_vertex_mesh(pts, tri_idx)
mesh.compute_triangle_normals()

# print(dir(mesh.triangle_normals))
# print(array(mesh.triangle_normals))

if __name__ == '__main__':

    # print(pts_2d)  # 对这个坐标三角化  三角化后得到三角形的顶点索引,根据这个索引从点云中检索,得到三维mesh的索引,构建mesh,求法向量

    ax = plt.figure(1).gca(projection='3d')

    # a = pts_buff
    ax.plot(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g.')
    # ax.plot(a[0], a[1], a[2], 'o')
    # ax.plot(x[0], x[1], x[2], 'r.')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.title('point cloud')
    plt.show()

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([mesh,
                                       axis_pcd
                                       ],
                                      window_name='ANTenna3D',
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      # point_show_normal=True
                                      )
