# ppf　特征计算 测试模块
#
# F(m1, m2) = <d, ang(n1, d), ang(n2, d), ang(n1, n2)>
# 要取整， 可以视为下采样  长度步长，角度步长

# 问题： 可否保证尺度不变？？

# 本代码使用三对点进行测试

import numpy as np
from numpy import *
from numpy.linalg import *
import open3d as o3d


a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

a_norm = array([1, 0, 0])
b_norm = array([0, 1, 0])


# 计算夹角
# 原理：向量内积的几何意义：向量a在b上的投影   返回弧度制夹角
def get_ang(a, b):
    theta = np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    return theta


def get_ppf(pt1, pt2, n1, n2):
    d = pt1 - pt2
    d_lenth = norm(d)
    d_unit = d / d_lenth
    # print('连线：', link_vector)

    alpha_n1_d = get_ang(n1, d)
    alpha_n2_d = get_ang(n2, d)
    alpha_n1_n2 = get_ang(a_norm, b_norm)

    ppf_vec = array([d_lenth, alpha_n1_d, alpha_n2_d, alpha_n1_n2])
    return ppf_vec


# 外积  可以将a写成反对陈矩阵，然后写成矩阵与向量b的乘法
# 几何意义：
#   两个向量张成四边形的有向面积，只对三维向量有意义
# c=axb
c = np.cross(a, b)
print(c)


if __name__ == '__main__':
    # 加载数据
    pcd = o3d.io.read_point_cloud('../imbalance_points(1)/imbalance_points/data_ply/Armadillo.ply')
    # pcd = o3d.io.read_point_cloud('data_ply/dragon_vrip.ply')
    pcd.paint_uniform_color([0.0, 0.5, 0.5])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    # print("Downsample the point cloud with a voxel of 0.05")
    pcd = pcd.voxel_down_sample(voxel_size=5)
    # print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    pcd_pt = array(pcd.points)
    pcd_norm = array(pcd.normals)

    pcd_trans = array(pcd.points)  # nx3
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pcd_trans = pcd_trans + array([130, 0, 0])
    # pcd_trans = pcd_trans + array([0.2, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

    # print("Recompute the normal of the downsampled point cloud")
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
    pcd2.paint_uniform_color([0.0, 0.8, 0.5])

    pcd_trans_normal = array(pcd2.normals)

    # 以索引进行保存

    # PPF参数设定
    d_step = 0.1
    a_step = 5

    num_pts = 1000
    for i in range(num_pts):
        pt_i = pcd_pt[i]
        pt_i_n = pcd_norm[i]
        for j in range(num_pts):
            pt_j = pcd_trans[j]
            pt_j_n = pcd_trans_normal[j]

            # 计算<mi, mj>的PPF

            ppf_vec = get_ppf(pt_i, pt_j, pt_i_n, pt_j_n)  # pt1, pt2, n1, n2

            # 特征离散化
            ppf_vec[0] = (ppf_vec[0] / d_step).astype(int)
            ppf_vec[1:] = (ppf_vec[1:] / a_step).astype(int)

    # 从构建pcd1 线下构建
    # pick one point
        # compare it with any other points
        # computer PPF
        # 离散化并向下取整 然后把点对索引聚类  点对特征相似的，聚集在一起
        # 保存

    # 全据特征描述是从PPF特征描述到模型的映射

    # 以pcd2作为场景 线上
        # 先选定一系列参考点
        # 其他所有的点和参考点 进行计算点对特征
        # 这些特征与模型的全局描述进行匹配？ 每一个潜在的匹配 投票位姿（相对于参考点）
        #

    # 场景中的所有点
    scene_pts = 1
    num_scene_pts = 1
    # 首先对场景进行采样，得到参考点
    scene_pts_r = 1
    num_pts_scene_r = 100

    for i in range(num_pts_scene_r):  # 参考点对其他所有点的特征
        pt_i = scene_pts_r[i]  # 参考点
        for j in range(num_scene_pts):
            pt_j = scene_pts[j]
            ppf_vec = get_ppf(pt_i, pt_j, pt_i_n, pt_j_n)  # pt1, pt2, n1, n2

            # 特征离散化
            ppf_vec[0] = (ppf_vec[0] / d_step).astype(int)
            ppf_vec[1:] = (ppf_vec[1:] / a_step).astype(int)

            # 通过查找哈希表得到 <mr, mi>，然后把<sr, si>和<mr, mi>统一到同一个坐标系

    o3d.visualization.draw_geometries([pcd,
                                       pcd2,
                                       axis_pcd
                                       ],
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      # point_show_normal=True
                                      )
