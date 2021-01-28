# ppf　特征计算模块
#
# F(m1, m2) = <d, ang(n1, d), ang(n2, d), ang(n1, n2)>
# 要取整， 可以视为下采样  长度步长，角度步长

# 问题： 可否保证尺度不变？？

import numpy as np
from numpy import *
from numpy.linalg import *
import open3d as o3d

from vec_pose import *

import time


# 计算夹角  注意 这个会导致不稳定，opencv建议使用 arctan?
# 原理：向量内积的几何意义：向量a在b上的投影
def get_ang(vec_1, vec_2):
    dist = dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))

    if dist > 1:  # 由于精度损失 可能会略微大于1 此时无解
        dist = 1

    theta = arccos(dist) * 180 / pi
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

train = 1

if __name__ == '__main__':
    # 加载数据
    # 模型
    pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
    # pcd = o3d.io.read_point_cloud('data_ply/dragon_vrip.ply')
    pcd.paint_uniform_color([0.0, 0.5, 0.5])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    pcd = pcd.voxel_down_sample(voxel_size=10)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    pcd_np = array(pcd.points)
    pcd_normal = array(pcd.normals)

    pts_num = len(pcd_np)  # 模型的点数

    # 制作场景  暂时只给定变换
    pcd_trans = array(pcd.points)  # nx3
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pcd_trans = pcd_trans + array([130, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

    # print("Recompute the normal of the downsampled point cloud")
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
    pcd2.paint_uniform_color([0.0, 0.8, 0.5])

    # 从构建pcd1 线下构建

    # compare it with any other points
    # computer PPF
    # 离散化并向下取整 然后把点对索引聚类  点对特征相似的，聚集在一起
    # 保存
    # 全据特征描述是从PPF特征描述到模型的映射

    # PPF参数设定
    d_step = 5
    a_step = 10

    # 训练
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
                    # 计算<mi, mj>的PPF  为了得到匹配的点
                    ppf_vec = get_ppf(pt_i, pt_j, pt_i_n, pt_j_n)  # pt1, pt2, n1, n2

                    # PPF离散化
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
                        hash_table[key_temp] = [value_temp]

            print(i / pts_num)  # 进度

        print(hash_table)
        save('hash_table.npy', hash_table)
        # 还要保存计算好的Alpha等参数

    # 以pcd2作为场景 线上匹配
    # 先选定一系列参考点,其他所有的点和参考点 进行计算点对特征
    # 这些特征与模型的全局描述进行匹配？ 每一个潜在的匹配 投票位姿（相对于参考点）

    # 加载hash table
    s_time = time.time()
    hash_table = load('hash_table.npy', allow_pickle=True).item()
    # print(hash_table)

    # 场景中的所有点
    scene_pts = array(pcd2.points)
    scene_pts_n = array(pcd2.normals)
    num_scene_pts = len(scene_pts)

    # 首先对场景进行采样，得到参考点  既然经过了下采样，其实不一定每个点都落在模型上
    # pcd2_r = pcd2.voxel_down_sample(voxel_size=15)
    pcd2_r = pcd2
    pcd2_r.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=10))

    scene_pts_r = array(pcd2_r.points)
    scene_pts_r_n = array(pcd2_r.normals)
    num_pts_scene_r = len(scene_pts_r)
    print('num_scene_pts:', num_scene_pts)
    print('num_pts_scene_r:', num_pts_scene_r)

    # 投票表
    a_col = int(360 / a_step)  # 如果遇到360度,直接使用之作为索引就不行 需要+1
    # print('a_col:', a_col)  #

    vote_table = zeros((num_pts_scene_r, a_col+1))  # 行：参考点个数  列：角度采样 0-360/step都包含
    pose_table = zeros((num_pts_scene_r, a_col+1, 3))  # 先只保存平移量，看出来的对不对，如果平移对了我再加上旋转。

    for i in range(num_pts_scene_r):  # 参考点对其他所有点的特征  注意 采样后参考点可能根本就不在模型上了
        pt_i = scene_pts_r[i]  # 参考点
        pt_i_n = scene_pts_r_n[i]
        for j in range(num_scene_pts):
            if i != j:  # 排除自己
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
                # value_temp = [i, j]

                # 将特征push到hash
                if key_temp in hash_table.keys():
                    # print('已经存在')
                    # 取出并进行匹配投票
                    match_pair = hash_table[key_temp]
                    # print('match_pair:', match_pair)

                    # 投票  这里哈系表已经完成了历史使命，接下来交给坐标变换
                    for pair in match_pair:
                        # 拿出每一个匹配到的点对
                        # 由于保存的是索引，所以需要从模型上查找到对应的点对信息
                        # 1.将mr与sr对齐到公共坐标系
                        mr = pcd_np[pair[0]]  # 模型参考点mr
                        mi = pcd_np[pair[1]]
                        mr_n = pcd_normal[pair[0]]

                        sr = pt_i  # 场景参考点sr
                        si = pt_j
                        sr_n = pt_i_n
                        # print(Tmg - Tsg)  # 开心的是，这里面已经能够检测出真实的平移变换

                        # 接下来就是求alpha
                        # 用到了<mr_n, mi, sr_n, si>

                        # 这里面的模型夹角可以放在前面，用来提速
                        # 后面+了180，这是在相对角度差上面加的，已经不是真实的了！
                        # 只是为了投票
                        alpha = get_alpha(mr, mi, mr_n, sr, si, sr_n) #+ 180  # 这一步太慢了! 先跑出来看看吧

                        alpha = int(alpha / a_step)  # 离散化 取整
                        # if alpha >= a_col:
                        #     alpha = a_col - 1

                        # print('alpha:', alpha)

                        # 投票
                        vote_table[i][alpha] += 1
                        pose_table[i][alpha] += mr - sr  # 每次得到的都是不一样的

        print('rate: {0:.3f}'.format(i / num_pts_scene_r))

    # print('vote_table:\n', vote_table)
    savetxt('vote_table.txt', vote_table)

    vote_max = np.max(vote_table)
    print('vote_max:\n', vote_max)

    vote_max_id = np.where(vote_max == vote_table)
    print('vote_max_id:', vote_max_id)  # (array([48], array([35])  alpha超过了

    x_id, y_id = vote_max_id[0], vote_max_id[1]  #
    print('pose_table_max:', pose_table[x_id][y_id] / vote_table[x_id][y_id])

    # 从table中找到可靠的局部坐标系
    # get pose
    # s_pose = get_pose()  # alpha_mi, rot_axis_mr_1, theta_mr_1, alpha_si, rot_axis_sr_1, theta_sr_1

    e_time = time.time()

    print('time cost: {0:.3f} s'.format(e_time - s_time))

    # 将模型变换,看结果是否正确

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
