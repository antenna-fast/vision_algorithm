from o3d_impl import *
from dist import *  # 距离计算
from np_lib import *

# 在mesh上面画出选中点以及2ring

mesh = read_mesh('D:/SIA/data_benchmark/mesh/Armadillo.ply', mesh_color=[0.5, 0.5, 0.5])

pcd_np = mesh2np(mesh)  # 用于索引

# 加载
pcd = o3d.io.read_point_cloud('D:/SIA/data_benchmark/mesh/Armadillo.ply')
pcd.paint_uniform_color([0.4, 0.4, 0.4])

pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)  # 构建搜索树

vici_num = 7
cut_num = 5

pts_num = len(pcd.points)

threshold = 0.5

i = 6992  # xiongxiong

idx_list = [6425, 355, 3174]

ring_buff_1 = []
ring_buff_2 = []

# 模型

# for i in range(pts_num):
# if 1:
for i in idx_list:
    # print("Paint the 1500th point red.")
    pick_idx = i

    # 一环上构造一个 特征向量
    [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
    vici_idx_1 = idx_1[1:]
    asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]  # 一环涂色
    asarray(pcd.colors)[pick_idx, :] = [1, 0, 0]

    ring_buff_1.append(vici_idx_1)

    now_pt_1 = array(pcd.points)[i]
    vici_pts_1 = array(pcd.points)[vici_idx_1]
    # all_pts = array(pcd.points)[idx_1]
    mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

    # 二环
    mesh_buff = []
    for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
        now_pt_1_2 = array(pcd.points[now_pt_2r])  # 每一个邻域的相对中心点

        # 搜索二环 邻域
        [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd.points[now_pt_2r], vici_num)
        vici_idx_1_2 = idx_1_2[1:]
        vici_pts_1_2 = array(pcd.points)[vici_idx_1_2]
        all_pts = array(pcd.points)[idx_1_2]

        for id in vici_idx_1_2:
            if id not in idx_1:  # 不要覆盖其他的点
                # id = vici_idx_1_2
                # if (id in vici_idx_1_2) and (id not in idx_1):  # 不要覆盖其他的点
                asarray(pcd.colors)[[id], :] = [0, 1, 0]  # 二环涂色  注意 这个二环又包含了中心点!
                ring_buff_2.append(id)

        mesh2, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)
        mesh_buff.append(mesh2)


ring_buff_1 = squeeze_vec(ring_buff_1)
ring_buff_2 = squeeze_vec(ring_buff_2)

cent_pts = pcd_np[idx_list]
ring_1 = pcd_np[ring_buff_1]
ring_2 = pcd_np[ring_buff_2]

size = 0.005
color_1 = [1, 0, 0]
color_2 = [0, 1, 0]
color_3 = [0, 0, 1]

cent_marker = keypoints_np_to_spheres(cent_pts, size=size, color=color_1)
ring_marker_1 = keypoints_np_to_spheres(ring_1, size=size, color=color_2)
ring_marker_2 = keypoints_np_to_spheres(ring_2, size=size, color=color_3)

o3d.visualization.draw_geometries([
    # pcd,
    mesh,
    cent_marker,
    ring_marker_1,
    ring_marker_2
    # pcd2,
    # axis_pcd,
],
    window_name='ANTenna3D',
    # zoom=0.3412,
    # front=[0.4257, -0.2125, -0.8795],
    # lookat=[2.6172, 2.0475, 1.532],
    # up=[-0.0694, -0.9768, 0.2024]
    # point_show_normal=True
)

# 添加
