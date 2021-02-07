from o3d_impl import *
from dist import *  # 距离计算
import os

import scipy.io as scio

# 加载数据集里的点云，加载GT，得到GT的feature并保存
# 在refine里面，根据特征进行筛选

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
model_name = 'rabbit'

# 模型路径
data_path = out_path + model_name + '.ply'
pcd = o3d.io.read_point_cloud(data_path)
print(pcd)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
pcd.paint_uniform_color([0.0, 0.6, 0.0])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)


vici_num = 7
cut_num = 5

pts_num = len(pcd.points)

threshold = 0.05  # 这里就用不到了


# 加载GT
gt_root = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/OUTPUT_DATA/GROUND_TRUTH_B/'
gt_path = gt_root + model_name + '.mat'
gt_idx_dic = scio.loadmat(gt_path)
print(gt_idx_dic.keys())
print(gt_idx_dic['GT_MODEL'].shape)

# T x N-1 x 2
sigma = 0.03
i = 3  # T sigma的个数?
n = 8
gt_idx = gt_idx_dic['GT_MODEL'][i, n-1, 0]  # 索引小1
print(gt_idx.shape)
gt_idx = squeeze(gt_idx) - 1  # 注意 要-1  matlab从1开始
print(gt_idx)


# 模型1
key_pts_buff_1 = []

for i in gt_idx:
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
    # 中心环的向量
    n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num]  # 规定长度
    # print(n_fn_angle_1.shape)  # len=cut_num

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

        n_fn_angle = []  # 清空单个周围环向量
        for f_normal in mesh_normals:
            # ang = arccos(dot(f_normal, vtx_normal) / (linalg.norm(f_normal) * linalg.norm(vtx_normal)))
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle.append(ang)

        # 周围环的所有向量
        # print('angle:', n_fn_angle)
        n_fn_angle = sort(array(n_fn_angle))[:cut_num]  # 预处理 排序+裁剪  固定长度
        n_fn_angle_buff_1.append(n_fn_angle)

    n_fn_angle_buff_1 = array(n_fn_angle_buff_1)
    # print(n_fn_angle_buff_1.shape)
    # 如何应对邻域个数不一致问题？如何匹配
    # 保存这些向量，作为局部特征被描述
    save_vec = vstack((n_fn_angle_1, n_fn_angle_buff_1))
    # print(save_vec.shape)  # 每一个关键点，都有7x4的特征

    f_name = 'feature_bag/' + model_name + '/' + model_name + '_gt_' + str(i) + '.txt'  # 所以这些保存的都是0开头的
    savetxt(f_name, save_vec)

    # print(n_fn_angle_buff_1)
    for vic_ang_1 in n_fn_angle_buff_1:
        # kl
        # print(len(vic_ang_1))
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

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   # pcd2,
                                   axis_pcd,
                                   ],
                                  window_name='ANTenna3D',
                                  )
