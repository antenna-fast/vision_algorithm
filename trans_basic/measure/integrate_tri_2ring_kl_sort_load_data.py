from o3d_impl import *
from dist import *  # 距离计算
import os

import scipy.io as io

# 加载数据集里的点云，检测后保存顶点索引
# 在matlab里面进行可视化（有mesh）

# 加载 1
out_path = 'D:/SIA/data_benchmark/'
file_list = os.listdir(out_path)  # 所有的模型

# for d_name in file_list:
#     data_path = out_path + d_name
# data_path = out_path + 'bird_3.ply'
# model_name = 'ant'  # 手动实验：为了得到测试结果，不太好批量化
# model_name = 'bird_3'
# model_name = 'armadillo'
# model_name = 'bust'
# model_name = 'girl'
# model_name = 'hand_3'
# model_name = 'camel'
# model_name = 'teddy'
# model_name = 'table_2'
# model_name = 'rabbit'
# model_name = 'octopus'
# model_name = 'teapot_2'
# model_name = 'glasses'
# model_name = 'dog_2'
# model_name = 'chair_4'
# model_name = 'chair_5'
# model_name = 'cow'
# model_name = 'cactus'
# model_name = 'cup'
# model_name = 'airplane_4'
# model_name = 'fish'
model_name = 'bird_2'


data_path = out_path + model_name + '.ply'
pcd = o3d.io.read_point_cloud(data_path)

print(pcd)
# pcd = pcd.voxel_down_sample(voxel_size=0.02)  # 不能下采样  否则改变索引
# print(pcd)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=10))
# pcd.paint_uniform_color([0.0, 0.5, 0.1])
pcd.paint_uniform_color([0.0, 0.6, 0.0])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)


# 加噪声
# noise_rate = 0.07  # 噪声占比  不同的噪声占比生成不同的关键点文件

vici_num = 7
cut_num = 4

pts_num = len(pcd.points)

# threshold = 0.5  # ant
# threshold = 1.9  # 数大 点少  camel
# threshold = 1.9  # girl  not use
threshold = 0.8  #

# i = 150
# 模型1
key_pts_buff_1 = []
for i in range(pts_num):
# if 1:
    pick_idx = i

    # 一环上构造一个特征向量
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
    vici_idx_1 = idx_1[1:]
    # asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]

    now_pt_1 = array(pcd.points)[i]
    vici_pts_1 = array(pcd.points)[vici_idx_1]
    # all_pts = array(pcd.points)[idx_1]
    mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

    # 构建一环的特征
    n_fn_angle_1 = []
    for f_normal in mesh_normals:
        ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
        n_fn_angle_1.append(ang)
        # print(ang)

    n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num]  # 规定长度
    # print(n_fn_angle_1)

    # 二环
    kl_buff = []
    n_fn_angle_buff_1 = []

    for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
        now_pt_1_2 = array(pcd.points[now_pt_2r])  # 每一个邻域的相对中心点

        # 搜索二环 邻域
        [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[now_pt_2r], vici_num)
        vici_idx_1_2 = idx_1_2[1:]
        vici_pts_1_2 = array(pcd.points)[vici_idx_1_2]
        all_pts = array(pcd.points)[idx_1_2]

        mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)

        n_fn_angle = []
        for f_normal in mesh_normals:
            # ang = arccos(dot(f_normal, vtx_normal) / (linalg.norm(f_normal) * linalg.norm(vtx_normal)))
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle.append(ang)

        # print('angle:', n_fn_angle)
        n_fn_angle_buff_1.append(n_fn_angle)

    # print(n_fn_angle_buff_1)
    for vic_ang_1 in n_fn_angle_buff_1:
        # kl
        # print(len(vic_ang_1))
        vic_ang_1 = sort(vic_ang_1)[:cut_num]  # 规定长度
        kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num)  # vec1, vec2, vec_len

        # print(kl_loss)
        kl_buff.append(kl_loss)

    kl_buff = array(kl_buff)
    # sum_var = var(var_buff)
    res = get_unbalance(kl_buff, threshold)  # 不平衡点

    if res:
        pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点
        # key_pts_buff_1.append(now_pt_1)  # 关键点
        key_pts_buff_1.append(i)  # 索引

print('key_pts_num:', len(key_pts_buff_1))

savetxt('save_file_kpt_idx/key_pts_buff_idx_' + model_name + '.txt', key_pts_buff_1, fmt='%d')
# savetxt('save_file_kpt_idx/key_pts_buff_idx_' + model_name + '.txt', key_pts_buff_1, fmt='%d')

# a = array([1, 2, 3, 4])
key_pts_buff_1 = array(key_pts_buff_1) + 1  # matlab的索引从1开始
s = {'IP_vertex_indices': key_pts_buff_1}
save_path = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours/' + model_name + '.mat'
io.savemat(save_path, s)

#
# # 聚类看看
# from sklearn.cluster import DBSCAN
#
# # Compute DBSCAN
# db = DBSCAN(eps=1, min_samples=3).fit(key_pts_buff_1)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print('labels:', labels)
# unique_labels = set(labels)  # 列表变成集合
# print(unique_labels)
#

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   # pcd2,
                                   axis_pcd,
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
