# ppf　特征计算模块
#
# F(m1, m2) = <d, ang(n1, d), ang(n2, d), ang(n1, n2)>
# 要取整， 可以视为下采样  长度步长，角度步长

# 问题： 可否保证尺度不变？？

import numpy as np
from numpy import *
from numpy.linalg import *
import open3d as o3d

a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

a_norm = array([1, 0, 0])
b_norm = array([0, 1, 0])


# 计算夹角  注意 这个会导致不稳定，opencv建议使用 arctan！
# 原理：向量内积的几何意义：向量a在b上的投影
def get_ang(vec_1, vec_2):
    theta = arccos(dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))) * 180 / pi
    return theta


def get_ppf(pt1, pt2, n1, n2):
    d = pt1 - pt2
    d_lenth = norm(d)
    # d_unit = d / d_lenth  # 单位化 顶点连接向量
    # print('连线：', link_vector)

    alpha_n1_d = get_ang(n1, d)
    alpha_n2_d = get_ang(n2, d)
    alpha_n1_n2 = get_ang(n1, n2)

    ppf_vec = array([d_lenth, alpha_n1_d, alpha_n2_d, alpha_n1_n2])
    return ppf_vec


# # 外积  可以将a写成反对陈矩阵，然后写成矩阵与向量b的乘法
# # 几何意义：
# #   两个向量张成四边形的有向面积，只对三维向量有意义
# # c=axb
# c = np.cross(a, b)
# print(c)

train = 0

if __name__ == '__main__':
    # 加载数据
    # 模型
    pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
    # pcd = o3d.io.read_point_cloud('data_ply/dragon_vrip.ply')
    pcd.paint_uniform_color([0.0, 0.5, 0.5])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    # print("Downsample the point cloud with a voxel of 0.05")
    pcd = pcd.voxel_down_sample(voxel_size=10)
    # print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    pcd_np = array(pcd.points)
    pcd_normal = array(pcd.normals)

    pts_num = len(pcd_np)  # 模型的点数


    # 制作场景
    pcd_trans = array(pcd.points)  # nx3
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pcd_trans = pcd_trans + array([130, 0, 0])
    # pcd_trans = pcd_trans + array([0.2, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

    # print("Recompute the normal of the downsampled point cloud")
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
    pcd2.paint_uniform_color([0.0, 0.8, 0.5])

    # 以索引进行保存

    # 从构建pcd1 线下构建
    # pick one point
        # compare it with any other points
        # computer PPF
        # 离散化并向下取整 然后把点对索引聚类  点对特征相似的，聚集在一起
        # 保存

    # 全据特征描述是从PPF特征描述到模型的映射

    # PPF参数设定
    d_step = 5
    a_step = 10

    if train:
        # 建立哈系表
        hash_table = {}

        for i in range(pts_num):  # 视为模型上的参考点
            pt_i = pcd_np[i]
            pt_i_n = pcd_normal[i]
            for j in range(pts_num):  # 
                if j != i:  # 不同的点之间比较
                    pt_j = pcd_np[j]
                    pt_j_n = pcd_normal[j]

                    # print('n: {0}  {1}'.format(pt_i_n, pt_j_n))
                    # 计算<mi, mj>的PPF
                    ppf_vec = get_ppf(pt_i, pt_j, pt_i_n, pt_j_n)  # pt1, pt2, n1, n2

                    # 特征离散化
                    ppf_vec[0] = (ppf_vec[0] / d_step).astype(int)
                    ppf_vec[1:] = (ppf_vec[1:] / a_step).astype(int)

                    # print(ppf_vec)
                    key_temp = str(ppf_vec)
                    value_temp = [i, j]

                    # 将特征push到hash
                    if key_temp in hash_table.keys():
                        # print('已经存在')
                        hash_table[key_temp].append(value_temp)
                    else:
                        # print('尚不存在，需要新建')
                        hash_table[key_temp] = value_temp

            print(i / pts_num)  # 进度

        print(hash_table)
        save('hash_table.npy', hash_table)

    # 以pcd2作为场景 线上匹配
    # 先选定一系列参考点
    # 其他所有的点和参考点 进行计算点对特征
    # 这些特征与模型的全局描述进行匹配？ 每一个潜在的匹配 投票位姿（相对于参考点）

    # 加载hash table
    hash_table = load('hash_table.npy', allow_pickle=True).item()
    # print(hash_table)
    print(hash_table)

    # 场景中的所有点
    scene_pts = array(pcd2.points)
    scene_pts_n = array(pcd2.normals)
    num_scene_pts = len(scene_pts)

    # 首先对场景进行采样，得到参考点
    pcd2_r = pcd2.voxel_down_sample(voxel_size=10)
    pcd2_r.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=11, max_nn=10))

    scene_pts_r = array(pcd2_r.points)
    scene_pts_r_n = array(pcd2_r.normals)
    num_pts_scene_r = len(scene_pts_r)


    # 投票表
    vote_table = zeros((100, 100))  # 行：参考点个数  列：角度采样

    for i in range(num_pts_scene_r):  # 参考点对其他所有点的特征
        pt_i = scene_pts_r[i]  # 参考点
        pt_i_n = scene_pts_r_n[i]
        for j in range(num_scene_pts):
            pt_j = scene_pts[j]
            pt_j_n = scene_pts_n[i]
            ppf_vec = get_ppf(pt_i, pt_j, pt_i_n, pt_j_n)  # pt1, pt2, n1, n2

            # 特征离散化
            ppf_vec[0] = (ppf_vec[0] / d_step).astype(int)
            ppf_vec[1:] = (ppf_vec[1:] / a_step).astype(int)

            # 通过查找哈希表得到 <mr, mi>，然后把<sr, si>和<mr, mi>统一到同一个坐标系 得到Tmg Tsg
            # 以及Alpha  （mi与si之间的夹角）

            # print(ppf_vec)
            key_temp = str(ppf_vec)
            value_temp = [i, j]

            # 将特征push到hash
            if key_temp in hash_table.keys():
                print('已经存在')

                # 取出并进行匹配投票

                # hash_table[key_temp].append(value_temp)

            # else:
            #     print('尚不存在，需要新建')
            #     hash_table[key_temp] = value_temp

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
