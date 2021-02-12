from o3d_impl import *
from detector import *
from dist import *  # 距离计算
import os


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

# model_name = 'armadillo'
# model_name = 'bust'
# model_name = 'girl'
model_name = 'hand_3'
# model_name = 'camel'
# model_name = 'teddy'
# model_name = 'table_2'
# model_name = 'rabbit'

# 对了，主要是讨论不同参数的
vici_num_list = [5, 6, 7, 8, 9, 10, 11]
# vici_num_list = [11]


# threshold_list = []

# threshold = 0.5  # ant
# threshold = 1.9  # 数大 点少  camel
# threshold = 1.9  # girl  not use
# threshold = 2.5  # armadillo  大概100点
# threshold = 1.3  # rabbit  大概 点
# threshold = 2.6  # girl  大概 点
# threshold = 1.6  # rabbit  大概 点
threshold = 1.9  # bust  大概 点


print('model_name:', model_name)

for vici_num in vici_num_list:
    print('vici_num:', vici_num)
    cut_num = vici_num - 3  # 因为

    # 保存格式：直接保存所有的索引，每次检测都得到一个
    for noise_rate in noise_list:
        print('noise_rate:', noise_rate)
        mesh_root = 'D:/SIA/data_benchmark/mesh_add_noise/'
        mesh_dir = mesh_root + model_name + '/' + str(noise_rate) + '.ply'

        mesh = read_mesh(mesh_dir)

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
