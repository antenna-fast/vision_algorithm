from o3d_impl import *
from dist import *  # 距离计算
import os

import scipy.io as io

# 对不同模型、不同噪声的数据进行检测
# 保存检测到的关键点

# 命名规则：
# vici_num  noise_rate.txt

noise_list = [0, 0.01, 0.03, 0.05, 0.07, 0.1]
# noise_list = [0]  # 补充漏下的


data_root = 'D:/SIA/data_benchmark/mesh_add_noise/'
file_list = os.listdir(data_root)  # 数据集的所有的模型

# 删除其他格式的 (txt)  --- 所有的模型列表  根据已有的所有模型
for i in range(len(file_list)):
    f_name = file_list[i]
    if '.txt' in f_name:
        file_list[i] = ''
# print(file_list)

# model_list = ['bust', 'girl', 'hand_3', 'camel', 'teddy', 'table_2', 'rabbit']
model_list = file_list

model_name = 'armadillo'
# model_name = 'bust'
# model_name = 'girl'
# model_name = 'hand_3'
# model_name = 'camel'
# model_name = 'teddy'
# model_name = 'table_2'
# model_name = 'rabbit'

# 对了，主要是讨论不同参数的

# vici_num_list = [5, 6, 7, 8, 9, 10, 11]
vici_num_list = [5, 6, 8, 9, 10, 11]  # 漏下的

# vici_num = 7
# cut_num = vici_num - 3  # 因为

# threshold_list = []

# threshold = 0.5  # ant
# threshold = 1.9  # 数大 点少  camel
# threshold = 1.9  # girl  not use
# threshold = 2.9  # armadillo  大概100点
threshold = 2.5  # armadillo  大概100点


# 这个是在pcd上的！
# 要把mesh转成pcd再过来
def detect_2_ring_kl(pcd_in, threshold, vici_num, cut_num):
    pts_num = len(pcd_in.points)

    pcd_in.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
    pcd_in.paint_uniform_color([0.0, 0.6, 0.0])
    pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd_in)  # 构建搜索树

    key_pts_buff_1 = []
    for i in range(pts_num):
        pick_idx = i

        # 一环上构造一个特征向量
        [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd_in.points[pick_idx], vici_num)
        vici_idx_1 = idx_1[1:]

        now_pt_1 = array(pcd_in.points)[i]
        vici_pts_1 = array(pcd_in.points)[vici_idx_1]
        # all_pts = array(pcd.points)[idx_1]
        mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

        # 构建一环的特征
        n_fn_angle_1 = []
        for f_normal in mesh_normals:
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle_1.append(ang)

        n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num]  # 规定长度

        # 二环
        kl_buff = []
        n_fn_angle_buff_1 = []

        for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
            now_pt_1_2 = array(pcd_in.points[now_pt_2r])  # 每一个邻域的相对中心点

            # 搜索二环 邻域
            [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd_in.points[now_pt_2r], vici_num)
            vici_idx_1_2 = idx_1_2[1:]
            vici_pts_1_2 = array(pcd_in.points)[vici_idx_1_2]
            all_pts = array(pcd_in.points)[idx_1_2]

            mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)

            n_fn_angle = []
            for f_normal in mesh_normals:
                ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
                n_fn_angle.append(ang)

            n_fn_angle_buff_1.append(n_fn_angle)

        for vic_ang_1 in n_fn_angle_buff_1:
            # kl
            vic_ang_1 = sort(vic_ang_1)[:cut_num]  # 规定长度
            kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num)  # vec1, vec2, vec_len

            kl_buff.append(kl_loss)

        kl_buff = array(kl_buff)
        # sum_var = var(var_buff)
        res = get_unbalance(kl_buff, threshold)  # 不平衡点

        if res:
            pcd_in.colors[pick_idx] = [1, 0, 0]  # 选一个点
            # key_pts_buff_1.append(now_pt_1)  # 关键点
            key_pts_buff_1.append(pick_idx)  # 索引

    return key_pts_buff_1


print('model_name:', model_name)

for vici_num in vici_num_list:
    print('vici_num:', vici_num)
    cut_num = vici_num - 3  # 因为

    # 保存格式：直接保存所有的索引，每次检测都得到一个
    for noise_rate in noise_list:
        print(noise_rate)
        mesh_root = 'D:/SIA/data_benchmark/mesh_add_noise/'
        mesh_dir = mesh_root + model_name + '/' + str(noise_rate) + '.ply'

        mesh = o3d.io.read_triangle_mesh(mesh_dir)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.0, 0.6, 0.1])

        # 转pcd
        pcd = mesh2pcd(mesh)

        # 检测 得到索引
        # pcd_in, threshold, vici_num, cut_num
        key_pts_buff_1 = detect_2_ring_kl(pcd, threshold, vici_num, cut_num)  # 检测  返回索引

        # 将结果(每个列表 包含同一个模型不同参数下的噪声相应)保存到对应文件夹
        print('key_pts_num:', len(key_pts_buff_1))

        save_root = 'D:/SIA/data_benchmark/mesh_add_noise_save/' + model_name

        if not(os.path.exists(save_root)):
            os.mkdir(save_root)

        save_txt_dir = save_root + '/' + str(vici_num) + '_' + str(noise_rate) + '.txt'

        savetxt(save_txt_dir, key_pts_buff_1, fmt='%d')

# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# o3d.visualization.draw_geometries([
#     # pcd,
#                                    axis_pcd,
#                                    ],
#                                   window_name='ANTenna3D',
#                                   )
