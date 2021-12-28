from o3d_impl import *
from dist import *  # 距离计算

import matplotlib.pyplot as plt

# 加载 1
pcd = o3d.io.read_point_cloud('../../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.4, 0.4, 0.4])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

noise_mode = 1  # 0 for vstack and 1 for jatter
noise_rate = 0.01  # 噪声占比
scale_ratio = 1  # 尺度

# 加载 2
pcd_trans = array(pcd.points) * scale_ratio  # nx3

# 找到邻域
# 输入当前点 邻域
# 输出: mesh, mesh_normals, normal

vici_num = 9
cut_num = 5

pts_num = len(pcd.points)

threshold = 0.5

i = 6992  # xiongxiong
i = 3511  # 耳朵尖  ear
# i = 2314  # 脚尖
# i= 8805  # 鼻子  可以  nose
# i = 621  # 胯部
i = 5460  # Knee
# i = 1671  # Knee

import configparser  # 配置文件读取

cfg = configparser.ConfigParser()
cfg.read('config.conf')
idx_list = list(map(int, cfg['idx_list']['idx_list'].split(',')))
print(idx_list)

# TODO:字典数据读取
print()

# idx_list = [1671, 8805, 3511]
# part_map = {3511: 'Ear', 2314: 'Tiptoe', 8805: 'Nose', 621: '', 6992:'Belly', 5460:'Knee'}

# 模型1
angle_buff = []  # 每个环上的角度向量

# for i in range(pts_num):
if 1:
# for i in idx_list:
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

    savetxt('../save_file/now_pt.txt', now_pt_1)
    savetxt('../save_file/vici_pts.txt', vici_pts_1)

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
    angle_buff.append(n_fn_angle_1)

    # 二环
    kl_buff = []  # 每次都要清空
    n_fn_angle_buff_1 = []

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

        mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)

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

    # if sum_var > threshold:
    #     pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点
    #     key_pts_buff_1.append(now_pt_1)

# savetxt('save_file/key_pts_buff_1_' + str(noise_rate) + '.txt', key_pts_buff_1)

# 保存角度向量
# angle_buff = array(angle_buff)
# savetxt('../save_file/angle_buff_1_' + str(i) + '.txt', angle_buff)


# 可视化角度向量
x_num = len(angle_buff[0])
x_line = list(range(1, x_num + 1))
row_num = len(angle_buff)
# print('x_line:', x_line)

# 水平刻度
for y_line in arange(0, 1, 0.1):
    plt.hlines(y_line, 1, x_num, colors="gray", linestyles="dashed")  # x, ymin, ymax
    # plt.vlines(i, 0, 0.5, colors="r", linestyles="dashed")  # x, ymin, ymax

# 所有的数据
# marker_map = {0: 'o', 1: '*', 2: '^', 3: 'x', 4: 'v', 5: 's', 6: 'p', 7: '+'}
# color_map = {}
# for row in range(row_num):
#     plt.plot(x_line, angle_buff[row])
#     plt.scatter(x_line, angle_buff[row], marker=marker_map[row])

# x坐标轴设置整数
# plt.xticks(x_line)
#
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 13,
#          }
# plt.title(part_map[i], font1)
#
# plt.show()

# 可视化KL
# print('kl_buff:', kl_buff)
# # savetxt('../save_file/kl_buff_' + str(pick_idx) + '.txt', kl_buff)
# savetxt('../save_file/kl_buff_' + str(pick_idx) + '.txt', kl_buff)
#
# x_line = list(range(len(kl_buff)))
# plt.plot(x_line, kl_buff)
# plt.show()


# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])
# # 查看选的点是否正确
# o3d.visualization.draw_geometries([pcd,
#                                    # pcd2,
#                                    # axis_pcd,
#                                    # mesh1,
#                                    # mesh2
#                                    ],
#                                   window_name='ANTenna3D',
#                                   # zoom=0.3412,
#                                   # front=[0.4257, -0.2125, -0.8795],
#                                   # lookat=[2.6172, 2.0475, 1.532],
#                                   # up=[-0.0694, -0.9768, 0.2024]
#                                   # point_show_normal=True
#                                   )
