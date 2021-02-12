from o3d_impl import *
from dist import *  # 距离计算

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os

# 关键点用来评估检测算法对噪声的鲁棒性

# 给定两个检测出来的关键点，在经过变换之后，看第二个检测出来的是不是在邻域范围之内
# 如果是，就认为是重合的


# rep(X,Y)  如果Y中检测到X中也有的关键点
# (rep(X,Y) + ep(Y,X)) / 2


# 思路，把第一组里面的点分别放倒第二组里面，然后检测最近邻
# 这样只需要构建一个pcd文件即可，添加进去的点索引是最后一位
# dist_threshold 小于阈值就认为是重复
def get_repeate_rate_2(pcd_np_1, pcd_np_2, dist_threshold):  # 第一位是GT  第二位是待检测

    vici_num = 2  # 2近邻 包含他自己

    pcd1_num = len(pcd_np_1)
    pcd2_num = len(pcd_np_2)

    repeat_num = 0  # 重复点计数器
    # all_repeat = min(pcd1_num, pcd2_num)  # 取出较小的一组
    all_repeat = pcd1_num

    for pt_idx in range(pcd1_num):  # 在GT里面拿点
        pt_1 = pcd_np_1[pt_idx]

        # 将变换后的点添加到变换后的场景中
        pcd2_np_temp = pcd_np_2  # 每次都更新
        pcd2_np_temp = vstack((pcd2_np_temp, pt_1))  #
        # print('len(pcd2_np_temp):', len(pcd2_np_temp))

        # 找到最近邻点 看距离是否在范围内  其实这种方法也不需要提前知道变换
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd2_np_temp)

        # 构建搜索树
        pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)

        [k, idx, _] = pcd_tree_2.search_knn_vector_3d(pcd2.points[pcd2_num], vici_num)  # 最后一个点的近邻 pcd2_num 就是最后添加进去的索引
        vici_idx = idx[1:]

        vici_pts = array(pcd2.points)[vici_idx]

        dist = sqrt(sum((pt_1 - vici_pts) ** 2))
        # print(dist)

        if dist < dist_threshold:  # 距离阈值
            repeat_num += 1

        # print(pt_idx / all_repeat)

    repeat_rate = repeat_num / all_repeat
    # print(repeat_rate)

    return repeat_rate


if __name__ == '__main__':

    noise_list = [0.01, 0.03, 0.05, 0.07, 0.1]
    model_list = ['armadillo', 'rabbit', 'camel', 'girl', 'rabbit']
    # model_list = ['ant', 'armadillo', 'bird_3', 'bust', 'girl', 'hand_3', 'camel', 'teddy', 'table_2', 'rabbit']
    # temp list  # 没有完全检测完，所以暂时

    # 每个文件：不同vici_num的文件
    vici_num_list = [5, 6, 7, 8, 9, 10, 11]

    dist_threshold = 0.05  # 距离小于阈值，就认为重复
    data_root = 'D:/SIA/data_benchmark/'

    font_1 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 13,
              }

    for noise_rate in noise_list:
        # noise_rate = 0.05
        # noise_rate = 0.07
        # noise_rate = 0.1

        # 上层：不同噪声  作为legend

        repeat_rate_buff = {}  # 不同模型 所有邻域的重复率 集成 # 清空
        for model_name in model_list:
            # model_name = 'armadillo'

            vici_buff = []
            for vici_num in vici_num_list:  # 关键点索引
                # print('vici_num:', vici_num)
                # 加载保存的 1.对应的模型
                mesh_gt_path = data_root + 'mesh_add_noise/' + model_name + '/' + '0.ply'  # noise == 0
                mesh_noise_path = data_root + 'mesh_add_noise/' + model_name + '/' + str(noise_rate) + '.ply'  # others

                mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)
                mesh_noise = o3d.io.read_triangle_mesh(mesh_noise_path)
                # 2.关键点索引
                idx_gt_path = data_root + 'mesh_add_noise_save/' + model_name + '/' + str(vici_num) + '_' + str(0) + '.txt'
                idx_noise_path = data_root + 'mesh_add_noise_save/' + model_name + '/' + str(vici_num) + '_' +\
                                 str(noise_rate) + '.txt'

                idx_gt = loadtxt(idx_gt_path).astype('int')
                idx_noise = loadtxt(idx_noise_path).astype('int')

                # mesh转换成np
                key_pts_buff_1 = mesh2np(mesh_gt)[idx_gt]
                key_pts_buff_2 = mesh2np(mesh_noise)[idx_noise]
                # print('key_pts_buff_1:\n', key_pts_buff_1)

                # 比较  gt noise threshold
                ra = get_repeate_rate_2(key_pts_buff_1, key_pts_buff_2, dist_threshold)
                # print('重复率', ra)

                vici_buff.append(ra)  # 所有邻域下的重复率

            # print('vici_buff:', vici_buff)
            repeat_rate_buff[model_name] = vici_buff

        print(repeat_rate_buff)

        # 对所有的模型（或者噪声）进行保存
        save_root = 'D:/SIA/data_benchmark/mesh_noise_repeat_rate/'
        repeat_rate_path = save_root   # 所有模型 重复率的保存目录

        if not(os.path.exists(repeat_rate_path)):
            os.mkdir(repeat_rate_path)

        repeat_rate_path += (str(noise_rate) + '.npy')
        # savetxt(repeat_rate_path, repeat_rate_buff)
        save(repeat_rate_path, repeat_rate_buff)

        # 绘图
        fig, ax = plt.subplots()  # Create a figure and an axes.

        for m_name in model_list:
            data_rep = repeat_rate_buff[m_name]
            plt.plot(vici_num_list, data_rep, marker='o')

        plt.legend(model_list, prop=font_1)  # 示意

        ax.set_xlabel('K points', font_1)  # Add an x-label to the axes.
        ax.set_ylabel('Repeatability', font_1)  # Add a y-label to the axes.

        # 加上标尺
        for y in arange(0.3, 1.1, 0.1):
            plt.hlines(y, 5, 11, colors="", linestyles="dashed")

        save_fig_path = 'D:/SIA/data_benchmark/fig/' + str(dist_threshold) + '_' + str(noise_rate) + '.jpg'
        plt.savefig(save_fig_path)

        # plt.show()

