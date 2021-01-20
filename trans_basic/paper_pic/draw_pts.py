from numpy import *
from numpy.linalg import *
import open3d as o3d

from o3d_impl import *
from base_trans import *
from dist import *  # 距离计算

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 加载 1
pcd = o3d.io.read_point_cloud('../../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.4, 0.4, 0.4])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

# 变换  旋转后索引发生变化！
# R = pcd.get_rotation_matrix_from_xyz((0, np.pi, 0))
# pcd.rotate(R, center=(0, 0, 0))

noise_mode = 1  # 0 for vstack and 1 for jatter
noise_rate = 0.01  # 噪声占比
scale_ratio = 1  # 尺度

# 加载 2
pcd_trans = array(pcd.points) * scale_ratio  # nx3

# 找到邻域
# 输入当前点 邻域
# 输出: mesh, mesh_normals, normal

vici_num = 7
cut_num = 5

pts_num = len(pcd.points)

threshold = 0.5

i = 6992  # xiongxiong
# i = 3511  # 耳朵尖  ear
i = 2314  # 脚尖
# i = 8805  # 鼻子  可以  nose
# i = 621  # 胯部
# i = 5460  # Knee
# i = 1671  # Knee
# i = 1937  # Nose

import configparser  # 配置文件读取

cfg = configparser.ConfigParser()
cfg.read('config.conf')
idx_list = list(map(int, cfg['idx_list']['idx_list'].split(',')))
print(idx_list)


idx_list = [6115, 11223, 1937]
idx_list = [1203, 5456, 11226]
# TODO:字典数据读取

# part_map = {3511: 'Ear', 2314: 'Tiptoe', 8805: 'Nose', 621: '', 6992:'Belly', 5460:'Knee'}

# 模型1
angle_buff = []  # 每个环上的角度向量

# for i in range(pts_num):
# if 1:
for i in idx_list:
    # print("Paint the 1500th point red.")
    pick_idx = i

    # 一环上构造一个 特征向量
    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
    vici_idx_1 = idx_1[1:]
    asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]  # 一环涂色
    asarray(pcd.colors)[pick_idx, :] = [1, 0, 0]

    now_pt_1 = array(pcd.points)[i]
    vici_pts_1 = array(pcd.points)[vici_idx_1]
    # all_pts = array(pcd.points)[idx_1]
    mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

    # 法向量同相化  和一环的同相  为时已晚
    # ref_normal = mesh_normals[0]
    # for n_i in range(len(mesh_normals)):
    #     if dot(mesh_normals[n_i], ref_normal) < 0:
    #         mesh_normals[n_i] = -mesh_normals[n_i]

    # 构建一环的特征
    n_fn_angle_1 = []
    for f_normal in mesh_normals:
        ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
        n_fn_angle_1.append(ang)
        # print(ang)

    n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num]  # 规定长度
    # print(n_fn_angle_1)
    angle_buff.append(n_fn_angle_1)

    # 二环
    kl_buff = []  # 每次都要清空
    n_fn_angle_buff_1 = []

    mesh_buff = []
    for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
        now_pt_1_2 = array(pcd.points[now_pt_2r])  # 每一个邻域的相对中心点

        # 搜索二环 邻域
        [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[now_pt_2r], vici_num)
        vici_idx_1_2 = idx_1_2[1:]
        vici_pts_1_2 = array(pcd.points)[vici_idx_1_2]
        all_pts = array(pcd.points)[idx_1_2]

        for id in vici_idx_1_2:
            if id not in idx_1:  # 不要覆盖其他的点
                asarray(pcd.colors)[[id], :] = [0, 1, 0]  # 二环涂色  注意 这个二环又包含了中心点!

        mesh2, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)
        mesh_buff.append(mesh2)

        n_fn_angle = []
        for f_normal in mesh_normals:
            # ang = arccos(dot(f_normal, vtx_normal) / (linalg.norm(f_normal) * linalg.norm(vtx_normal)))
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle.append(ang)
            # print(ang)

        n_fn_angle = sort(array(n_fn_angle)[: cut_num])
        # print('angle:', n_fn_angle)
        n_fn_angle_buff_1.append(n_fn_angle)
        angle_buff.append(n_fn_angle)

    # print(n_fn_angle_buff_1)
    for vic_ang_1 in n_fn_angle_buff_1:
        # kl
        # print(len(vic_ang_1))
        vic_ang_1 = sort(vic_ang_1)  # 规定长度
        kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num)  # vec1, vec2, vec_len

        # print(kl_loss)
        kl_buff.append(kl_loss)

    kl_buff = array(kl_buff)
    # sum_var = var(var_buff)
    # 使用排序

o3d.visualization.draw_geometries([pcd,
                                   # pcd2,
                                   # axis_pcd,
                                   mesh1,
                                   mesh_buff[0],
                                   mesh_buff[1],
                                   mesh_buff[2],
                                   mesh_buff[3],
                                   mesh_buff[4],
                                   mesh_buff[5],
                                   # mesh_buff[6],

                                   ],
                                  window_name='ANTenna3D',
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 2.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]
                                  # point_show_normal=True
                                  )

# 添加
