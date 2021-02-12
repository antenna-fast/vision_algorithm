from numpy import *
import open3d as o3d
from o3d_impl import *
from dist import *
import scipy.io as scio

import os

# 约定：txt等python处理过的文件索引从0开始
# matlab的从1开始

# 读取检测的关键点
# 求关键点的特征矩阵
# 将此特征矩阵和GT的进行对比，如果相似度小于一个数值 就可以。
#   细节：结果不能受行向量的排列影响

# 读取索引，拿到coarse

# model_name = 'ant'
# model_name = 'bird_3'
# model_name = 'armadillo'
# model_name = 'bust'
# model_name = 'girl'
# model_name = 'hand_3'
# model_name = 'camel'
# model_name = 'teddy'
# model_name = 'rabbit'
# model_name = 'octopus'
# model_name = 'dog_2'
# model_name = 'cow'
# model_name = 'cactus'
# model_name = 'fish'
model_name = 'bird_2'

vici_num = 7
cut_num = 4

num_mat = vici_num * cut_num

# threshold_refine = 0.00009  # 如果小于0.5 则视为重复  ant
# threshold_refine = 0.00001  # 如果小于0.5 则视为重复  camel
# threshold_refine = 0.00001  # 如果小于0.5 则视为重复  girl
# threshold_refine = 0.00003  # 如果小于0.5 则视为重复  teddy
# threshold_refine = 0.00003  # 如果小于0.5 则视为重复  table_2
threshold_refine = 0.00001  # 如果小于0.5 则视为重复  rabbit


# 检测到的所有关键点加载（粗）
file_name = '../save_file_kpt_idx/key_pts_buff_idx_' + model_name + '.txt'
key_pts_idx = loadtxt(file_name, dtype='int')  # s0
# print(key_pts)

# 模型路径
out_path = 'D:/SIA/data_benchmark/'
data_path = out_path + model_name + '.ply'
pcd = o3d.io.read_point_cloud(data_path)
print(pcd)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
pcd.paint_uniform_color([0.0, 0.6, 0.0])
# asarray(pcd.colors)[key_pts, :] = [1, 0, 0]
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

pts_num = len(pcd.points)


# GT matrix
# 首先加载GT index
# 加载GT
gt_root = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/OUTPUT_DATA/GROUND_TRUTH_B/'
gt_path = gt_root + model_name + '.mat'
gt_idx_dic = scio.loadmat(gt_path)
# print(gt_idx_dic.keys())
# print('gt_idx_dic:', gt_idx_dic['GT_MODEL'].shape)

# T x N-1 x 2
sigma = 0.03
i = 3  # T sigma的个数
n = 11
gt_idx = gt_idx_dic['GT_MODEL'][i, n-1, 0]  # 索引小1
gt_idx = squeeze(gt_idx) - 1  # notice  这里是变成python计算的索引
# print(gt_idx.shape)

feature_bag_root = 'C:/Users/yaohua-win/PycharmProjects/pythonProject/vision_algorithm/trans_basic/measure/key_pts_refine/feature_bag/'
model_feature_path = feature_bag_root + model_name + '/'

# 这里已经是0开始 字典
gt_feature_mat_buff = {}  # 根据gt索引读取计算好的gt feature
for idx in gt_idx:
    gt_file_name = model_feature_path + model_name + '_gt_' + str(idx) + '.txt'
    gt_f_temp = loadtxt(gt_file_name)
    gt_feature_mat_buff[str(idx)] = gt_f_temp

iter_key = gt_feature_mat_buff.keys()  # GT特征 迭代列表

# print('gt_feature_mat_buff:\n', gt_feature_mat_buff)
# better way: 加载后保存到字典

# gt_f_path = gt_path + '' +
# gt_feature_mat = loadtxt(gt_path)


# 在粗糙关键点周围进行检验
key_pts_refine_buff = []

for i in key_pts_idx:  # 所有检测到的关键点  一旦有特征相似的，就保存
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
    feature_mat = vstack((n_fn_angle_1, n_fn_angle_buff_1))
    # print(save_vec.shape)  # 每一个关键点，都有7x4的特征

    # 遍历所有的GT矩阵，相似的保留
    for gt_iter in iter_key:
        g = gt_feature_mat_buff[gt_iter]  # 拿到特征矩阵
        # 比较
        # f_delta = feature_mat - g  # 直接做差无法得到行不变

        # res = abs(sum(f_delta))
        res = abs((var(g) - var(feature_mat))) / num_mat
        # print(res)
        if res < threshold_refine:
            # # 和已经有的进行比较，如果距离小于阈值就不再添加
            # for exist_idx in key_pts_refine_buff:  # 一开始是空的
            #     exist_pt = array(pcd.points)[exist_idx]
            #     dist_temp = get_Euclid(now_pt_1, exist_pt)
            #     if dist_temp < 0.1:
            #             key_pts_refine_buff.append(i)
            key_pts_refine_buff.append(i)

            break  # 如果已经匹配到，则此关键点不再继续寻找匹配
        # print(abs(sum(f_delta)))

key_pts_refine_buff = array(key_pts_refine_buff)
print(len(key_pts_refine_buff))

# 存储新的索引
key_pts_buff_1 = array(key_pts_refine_buff) + 1  # matlab的索引从1开始
s = {'IP_vertex_indices': key_pts_buff_1}
save_path = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours/' + model_name + '.mat'
scio.savemat(save_path, s)

    # refine之后的点 存放位置
    # f_name = 'feature_bag/ant/ant_gt_' + str(i) + '.txt'
    # savetxt(f_name, save_vec)

    # if res:
    #     pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点
    #     # key_pts_buff_1.append(now_pt_1)  # 关键点
    #     key_pts_refine_buff.append(i)  # 索引
#