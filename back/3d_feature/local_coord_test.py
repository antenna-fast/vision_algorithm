from numpy import *
import numpy as np
import open3d as o3d
from open3d import *
import time

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.spatial import Delaunay
import numpy as np

points = np.array([[0.1933333, 0.47],
                   [0.1966667, 0.405],
                   [0.2066667, 0.3375]])


tri = Delaunay(points)
print(tri)

delaunay_plot_2d(tri)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.show()


def get_normal(a):
    average_data = np.mean(a, axis=0)  # 求 NX3 向量的均值
    decentration_matrix = a - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # S

    sort = eigenvalues.argsort()[::-1]  # 降序排列
    eigenvalues = eigenvalues[sort]  # 索引
    eigenvectors = eigenvectors[:, sort]

    # print(eigenvectors[2])
    return eigenvectors[2]


# 问题：邻域点旋转后，法向量不一致

mean = array([0, 0, 0])
# mean = array([0, 0])
# cov = eye(3)
# cov = eye(2)
cov = array([[5, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])

a = random.multivariate_normal(mean, cov, 1000)
# print(a)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(a)
pcd.paint_uniform_color([1, 0.0, 0.0])
# print("Recompute the normal of the downsampled point cloud")
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd)
vis.add_geometry(axis_pcd)

# 对椭球旋转 观察最小特征值对应的特征向量
for i in arange(0, 10, 0.01):

    # 变换
    r_vec = R.from_rotvec(array([i, 0, 0]) * 180 / pi)
    # r_vec = R.from_rotvec(array([i, 0, 0]) * 180 / pi)
    r_mat = r_vec.as_matrix()
    print(r_mat)

    pcd_np = array(pcd.points)

    pcd_np_t = dot(r_mat, pcd_np.T).T

    pcd.points = o3d.utility.Vector3dVector(pcd_np_t)
    pcd.paint_uniform_color([1, 0.0, 0.0])
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    # 求法向量
    normal = get_normal(pcd_np_t)
    print(normal)

    # 赋值法向量

    # if i == 0:
    #     vis.add_geometry(pcd)

    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.2)

o3d.visualization.draw_geometries([pcd,
                                   # pts_2_o,
                                   axis_pcd
                                   ],
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 2.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]
                                  # point_show_normal=True
                                  )

# vis.destroy_window()
