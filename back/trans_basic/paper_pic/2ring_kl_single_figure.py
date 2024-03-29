
from o3d_impl import *
from base_trans import *
from dist import *  # 距离计算

import matplotlib.pyplot as plt

# 加载 1
pcd = o3d.io.read_point_cloud('../../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.5, 0.5])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

noise_mode = 1  # 0 for vstack and 1 for jatter
noise_rate = 0.01  # 噪声占比
scale_ratio = 1  # 尺度

# 加载 2
pcd_trans = array(pcd.points) * scale_ratio  # nx3

# 定义变换
r = R.from_rotvec(pi / 180 * array([30, 60, 30]))  # 角度->弧度
# r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([150, -2, -8], dtype='float')
print('r_mat:\n', r_mat)

pcd_trans = dot(r_mat, pcd_trans.T).T
pcd_trans = pcd_trans + t_vect

# 加噪声

noise_mode = 2  # 0 for vstack and 1 for jatter

noise_rate = 0.07  # 噪声占比

if noise_mode == 0:
    mean = array([0, 0, 0])
    cov = eye(3) * 1000
    # cov = eye(3)*diameter  # 直径的多少倍率

    pts_num = len(pcd_trans)
    noise_pts_num = int(pts_num * noise_rate)
    noise = random.multivariate_normal(mean, cov, noise_pts_num)
    # 对噪声变换到场景坐标系
    noise_trans = dot(r_mat, noise.T).T + t_vect
    # 方式1 将噪声塞进去
    pcd_trans = vstack((pcd_trans, noise_trans))

if noise_mode == 1:
    mean = array([0, 0, 0])
    cov = eye(3)  # 直径的多少倍率

    pts_num = len(pcd_trans)
    noise_pts_num = int(pts_num * noise_rate)
    noise = random.multivariate_normal(mean, cov, noise_pts_num)
    # print('noise.shape:', noise.shape)

    # 对噪声变换到场景坐标系
    noise_trans = dot(r_mat, noise.T).T + t_vect

    # 方式2 对模型点跳动
    # 第二种 不改变点的整体数量,直接对采样的点添加
    rand_choose = np.random.randint(0, pts_num, noise_pts_num)
    pcd_trans[rand_choose] += noise

else:
    pass

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.5, 0.1])
# pcd_trans_normal = array(pcd2.normals)

# 构建搜索树
pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)

print('pcd1_num:', len(pcd.points))
print('pcd2_num:', len(pcd2.points))

# 可视化待检测数据


# 找到邻域
# 输入当前点 邻域
# 输出: mesh, mesh_normals, normal
histo = {}

vici_num = 7
cut_num = 5

pts_num = len(pcd.points)

threshold = 0.5

i = 6992  # xiongxiong
# i = 3511  # 耳朵尖  ear  6115
# i = 2314  # 脚尖
i = 8805  # 鼻子  可以  nose  1937
# i = 621  # 胯部
# i = 5460  # test
i = 1671  # test
# i = 1343  # test
i = 11223  # Knee
# i = 6115  # 耳朵
# i = 8597

# idx_list = [1671, 8805, 3511]  # 耳朵和鼻子很像
idx_list = [i]  # 耳朵和鼻子很像
part_map = {6115: 'Ear', 2314: 'Tiptoe', 1937: 'Nose', 621: '', 8597: 'Belly',
            11223: 'Knee',
            # i: 'Test'
            }

# 模型1

# kl_buff = []

# for i in range(pts_num):
# if 1:
for i in idx_list:

    angle_buff = []  # 每个环上的角度向量

    # print("Paint the 1500th point red.")
    pick_idx = i

    # 一环上构造一个 特征向量
    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
    vici_idx_1 = idx_1[1:]
    asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]

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
    n_fn_angle_1 = n_fn_angle_1/sum(n_fn_angle_1)
    for vic_ang_1 in n_fn_angle_buff_1:
        # kl
        # print(len(vic_ang_1))
        # vic_ang_1 = sort(vic_ang_1)  # 规定长度
        vic_ang_1 = vic_ang_1/sum(vic_ang_1)
        # print('vic_ang_1:', vic_ang_1)
        # print('n_fn_angle_1', n_fn_angle_1)
        kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num)  # vec1, vec2, vec_len

        # print(kl_loss)
        kl_buff.append(kl_loss)
    kl_buff = sort(kl_buff)
    print('kl_buff:', kl_buff)
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
    print('x_line:', x_line)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    # 水平刻度
    for y_line in arange(0, 1, 0.1):
        plt.hlines(y_line, 1, x_num, colors="gray", linestyles="dashed")  # x, ymin, ymax
        # plt.vlines(i, 0, 0.5, colors="r", linestyles="dashed")  # x, ymin, ymax

    # 所有的数据
    marker_map = {0: 'o', 1: '*', 2: '^', 3: 'x', 4: 'v', 5: 's', 6: 'p', 7: '+'}
    color_map = {}
    for row in range(row_num):
        plt.plot(x_line, angle_buff[row], label=str(row) + ' ring')
        plt.scatter(x_line, angle_buff[row], marker=marker_map[row])

    # x坐标轴设置整数
    plt.xticks(x_line)

    plt.title(part_map[i], font1)
    # plt.axis('off')  # 关闭坐标轴
    plt.legend(prop=font1, loc=4)  # 设置legend的字体
    plt.show()

    # 可视化KL
    # print('kl_buff:', kl_buff)
    # # savetxt('../save_file/kl_buff_' + str(pick_idx) + '.txt', kl_buff)
    savetxt('../save_file/kl_buff_' + str(pick_idx) + '.txt', kl_buff)
    #
    # x_line = list(range(len(kl_buff)))
    # plt.plot(x_line, kl_buff)
    #
    # plt.show()

