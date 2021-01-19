from numpy import *
from numpy.linalg import *
import open3d as o3d

from base_trans import *
from dist import *  # 距离计算

from o3d_impl import *

# 超参数
search_radius = 0.1
search_max_nn = 10

# 下采样
voxel_1 = 0.005
voxel_2 = 0.007

# 加载 1
model_path = '../data_ply/bun_zipper.ply'

pcd = o3d.io.read_point_cloud(model_path)
pcd = pcd.voxel_down_sample(voxel_size=voxel_1)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=search_max_nn))
pcd.paint_uniform_color([0.0, 0.5, 0.1])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

# 加载 2
scene_path = '../data_ply/Scene6_1-8.ply'

pcd2 = o3d.io.read_point_cloud(model_path)
pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_2)

pcd_trans = array(pcd2.points)  # nx3

# 定义变换  评估的时候要给定
r = R.from_rotvec(pi / 180 * array([0, 30, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([0.4, 0, 0], dtype='float')
# print('r_mat:\n', r_mat)

# pcd_trans = dot(r_mat, pcd_trans.T).T
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
pcd2.paint_uniform_color([0.0, 0.5, 0.1])

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=search_max_nn))
pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)  # 构建搜索树

print('pcd1_num:', len(pcd.points))
print('pcd2_num:', len(pcd2.points))

# 输入当前点 邻域
# 输出: mesh, mesh_normals, normal
histo = {}


# 邻域点个数
vici_num_1 = 7
vici_num_2 = 8  # 稀疏的点使用更大的邻域？  自适应：基于密度  选取半径内的点，计算点的个数
# 排序后截取的个数
cut_num_1 = 5
cut_num_2 = 4

pts_num_m = len(pcd.points)
pts_num_s = len(pcd2.points)

threshold = 1

# i = 150
# 模型1
key_pts_buff_1 = []
for i in range(pts_num_m):
    # if 1:
    # print("Paint the 1500th point red.")
    pick_idx = i

    # 一环上构造一个 特征向量
    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num_1)
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

    n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num_1]  # 规定长度
    # print(n_fn_angle_1)

    # 二环
    var_buff = []
    n_fn_angle_buff_1 = []

    for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
        now_pt_1_2 = array(pcd.points[now_pt_2r])  # 每一个邻域的相对中心点

        # 搜索二环 邻域
        [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[now_pt_2r], vici_num_1)
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

        # print('angle:', n_fn_angle)
        n_fn_angle_buff_1.append(n_fn_angle)

    # print(n_fn_angle_buff_1)
    for vic_ang_1 in n_fn_angle_buff_1:
        # kl
        # print(len(vic_ang_1))
        vic_ang_1 = sort(vic_ang_1)[:cut_num_1]  # 规定长度
        kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num_1)  # vec1, vec2, vec_len

        # print(kl_loss)
        var_buff.append(kl_loss)

    var_buff = array(var_buff)
    sum_var = var(var_buff)

    if sum_var > threshold:
        pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点
        key_pts_buff_1.append(now_pt_1)

savetxt('save_file/key_pts_buff_1_' + str(noise_rate) + '.txt', key_pts_buff_1)

# 变换后  模型2
# if 1:
key_pts_buff_2 = []
for i in range(pts_num_s):

    pick_idx = i

    # 一环上构造一个 特征向量
    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_2, _] = pcd_tree_2.search_knn_vector_3d(pcd2.points[pick_idx], vici_num_2)
    vici_idx_2 = idx_2[1:]
    # asarray(pcd.colors)[vici_idx_2, :] = [0, 0, 1]

    now_pt_2 = array(pcd2.points)[i]
    vici_pts_2 = array(pcd2.points)[vici_idx_2]
    # all_pts = array(pcd2.points)[idx_2]
    mesh2, mesh_normals, vtx_normal = get_mesh(now_pt_2, vici_pts_2)

    # 构建一环的特征
    n_fn_angle_2 = []
    for f_normal in mesh_normals:
        ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
        n_fn_angle_2.append(ang)

    n_fn_angle_2 = sort(array(n_fn_angle_2))[:cut_num_2]  # 规定长度   先排序
    # print(n_fn_angle_2)

    # 二环
    var_buff = []
    n_fn_angle_buff_2 = []

    for now_pt_2r in vici_idx_2:  # 比较二环与中心环的区别
        now_pt_2_2 = array(pcd2.points[now_pt_2r])  # 每一个邻域的相对中心点

        # 搜索二环 邻域
        [k, idx_2_2, _] = pcd_tree_2.search_knn_vector_3d(pcd2.points[now_pt_2r], vici_num_2)
        vici_idx_2_2 = idx_2_2[1:]
        vici_pts_2_2 = array(pcd2.points)[vici_idx_2_2]

        mesh2, mesh_normals, vtx_normal = get_mesh(now_pt_2_2, vici_pts_2_2)

        # 二环角度向量
        n_fn_angle = []
        for f_normal in mesh_normals:
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle.append(ang)
            # print(ang)
        # print('angle:', n_fn_angle)
        n_fn_angle_buff_2.append(n_fn_angle)

    for vic_ang_2 in n_fn_angle_buff_2:
        # kl
        vic_ang_2 = sort(vic_ang_2)[:cut_num_2]  # 规定长度
        kl_loss = get_KL(vic_ang_2, n_fn_angle_2, cut_num_2)  # vec1, vec2, vec_len

        var_buff.append(kl_loss)

    var_buff = array(var_buff)
    sum_var = var(var_buff)
    # print('sum_var2:', sum_var)

    if sum_var > threshold:
        pcd2.colors[pick_idx] = [1, 0, 0]  # 选一个点
        key_pts_buff_2.append(now_pt_2)
key_pts_buff_2 = array(key_pts_buff_2)
savetxt('save_file/key_pts_buff_2_' + str(noise_rate) + '.txt', key_pts_buff_2)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   pcd2,
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
