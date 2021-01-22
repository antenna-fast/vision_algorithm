from dist import *  # 距离计算
from o3d_impl import *

# TODO:将度量改成KL散度 + 方差

# 加载 1
pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=1)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.5, 0.1])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

# 加载 2
pcd_trans = array(pcd.points)  # nx3

# 计算点云直径
dx = pcd_trans[:, 0].max() - pcd_trans[:, 0].min()
dy = pcd_trans[:, 1].max() - pcd_trans[:, 1].min()
dz = pcd_trans[:, 2].max() - pcd_trans[:, 2].min()
diameter = sqrt(norm([dx ** 2, dy ** 2, dz ** 2]))
# print(dx, dy, dz)
print(diameter)

# 定义变换
r = R.from_rotvec(pi / 180 * array([30, 60, 30]))  # 角度->弧度
# r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([150, -2, -8], dtype='float')
print('r_mat:\n', r_mat)

pcd_trans = dot(r_mat, pcd_trans.T).T
pcd_trans = pcd_trans + t_vect

# 加噪声

noise_mode = 2  # 0 for vstack and 1 for jitter

noise_rate = 0.1  # 噪声占比

mean = array([1, 0, 1])

if noise_mode == 0:
    cov = eye(3) * 1000
    # cov = eye(3)*diameter  # 直径的多少倍率

    pts_num = len(pcd_trans)
    noise_pts_num = int(pts_num * noise_rate)
    noise = random.multivariate_normal(mean, cov, noise_pts_num)
    # print('noise.shape:', noise.shape)
    # 对噪声变换到场景坐标系

    noise_trans = dot(r_mat, noise.T).T + t_vect

if noise_mode == 1:
    cov = eye(3)   # 直径的多少倍率

    pts_num = len(pcd_trans)
    noise_pts_num = int(pts_num * noise_rate)
    noise = random.multivariate_normal(mean, cov, noise_pts_num)
    # print('noise.shape:', noise.shape)

    # 对噪声变换到场景坐标系

    noise_trans = dot(r_mat, noise.T).T + t_vect

if noise_mode == 0:
    # 方式1 将噪声塞进去
    pcd_trans = vstack((pcd_trans, noise_trans))

if noise_mode == 1:
    # 方式2 对模型点跳动
    # 第二种 不改变点的整体数量,直接对采样的点添加
    rand_choose = np.random.randint(0, pts_num, noise_pts_num)
    pcd_trans[rand_choose] += noise

# 不加噪声
else:
    pass

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

print('pcd1_num:', len(pcd.points))
print('pcd2_num:', len(pcd2.points))

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.5, 0.1])
# pcd_trans_normal = array(pcd2.normals)

# 构建搜索树
pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

# 遍历

# 找到邻域
# 输入当前点 邻域
# 输出: mesh, mesh_normals, normal
histo = {}


vici_num = 9

pts_num_1 = len(pcd.points)

# i = 1500
# 模型1
for i in range(pts_num_1):
# for i in range(1):
    # print("Paint the 1500th point red.")
    pick_idx = i
    now_pt_1 = array(pcd.points[pick_idx])
    # pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点

    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
    vici_idx_1 = idx_1[1:]
    # asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]

    vici_pts_1 = array(pcd.points)[vici_idx_1]
    all_pts = array(pcd.points)[idx_1]

    mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)
    # print('mesh_normals:\n', mesh_normals)
    # print('mesh2:', array(mesh1.vertices))

    n_fn_angle = []
    for f_normal in mesh_normals:

        # ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
        ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
        n_fn_angle.append(ang)
        # print(ang)

    n_fn_angle = sort(n_fn_angle)[:8]
    n_fn_angle = (array(n_fn_angle)+1) / 2
    # print('angle:', n_fn_angle)

    # 计算mesh法向量的离散程度   # 统计余弦值的方差
    var_cos = var(n_fn_angle, axis=0)
    var_cos = sum(var_cos)

    # 如果当前邻域的离散程度超过阈值，就标记出
    if var_cos > 0.05:
        pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点

# print(histo)
pts_num_2 = len(pcd2.points)

# 变换后
for i in range(pts_num_2):
    # if 1:
    # print("Paint the 1500th point red.")
    pick_idx = i
    now_pt_2 = array(pcd2.points[pick_idx])
    # pcd2.colors[pick_idx] = [1, 0, 0]  # 选一个点

    # print("Find its nearest neighbors, and paint them blue.")
    [k, idx_2, _] = pcd_tree2.search_knn_vector_3d(pcd2.points[pick_idx], vici_num)
    vici_idx_2 = idx_2[1:]
    # asarray(pcd2.colors)[vici_idx_2, :] = [0, 0, 1]

    vici_pts_2 = array(pcd2.points)[vici_idx_2]
    all_pts = array(pcd2.points)[idx_2]
    # print(vici_pts_2)

    mesh2, mesh_normals2, vtx_normal2 = get_mesh(now_pt_2, vici_pts_2)
    # print('mesh_normals2:\n', mesh_normals2)
    # print('mesh2:', mesh2)
    # print('mesh2:', array(mesh2.vertices))

    n_fn_angle = []
    for f_normal in mesh_normals2:
        # ang = arccos(dot(f_normal, pt_normal) / (linalg.norm(f_normal) * linalg.norm(pt_normal)))
        # ang = (dot(f_normal, vtx_normal2) / (linalg.norm(f_normal) * linalg.norm(vtx_normal2)))
        ang = get_cos_dist(f_normal, vtx_normal2)
        n_fn_angle.append(ang)
    n_fn_angle = array(n_fn_angle)

    print('angle2:', n_fn_angle)

    # 计算mesh法向量的离散程度   # 统计余弦值的方差
    var_cos = var(n_fn_angle, axis=0)
    var_cos = sum(var_cos)

    if var_cos > 0.05:
        pcd2.colors[pick_idx] = [1, 0, 0]  # 选一个点

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   pcd2,
                                   axis_pcd,
                                   mesh1,
                                   mesh2
                                   ],
                                  window_name='ANTenna3D',
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 2.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]
                                  # point_show_normal=True
                                  )
