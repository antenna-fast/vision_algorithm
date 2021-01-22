from o3d_impl import *

# 加载 1
pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# 构建搜索树
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# 加载 2
pcd2 = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd2 = pcd2.voxel_down_sample(voxel_size=2)
pcd2.paint_uniform_color([0.0, 0.5, 0.5])

pcd_trans = array(pcd.points)  # nx3

# 定义变换
r = R.from_rotvec(pi / 180 * array([0, 20, 10]))  # 角度->弧度
# r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([150, -2, -8], dtype='float')
print('r_mat:\n', r_mat)

# pcd_trans = dot(r_mat, pcd_trans.T).T
pcd_trans = pcd_trans + t_vect

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.8, 0.5])

pcd_trans_normal = array(pcd2.normals)

# 构建搜索树
pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)


# 遍历

vici_num = 9

print("Paint the 1500th point red.")
pick_idx = 1500
now_pt = array(pcd.points[pick_idx])

pcd.colors[pick_idx] = [1, 0, 0]  # 选一个点

print("Find its nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[pick_idx], vici_num)
vici_idx = idx[1:]
asarray(pcd.colors)[vici_idx, :] = [0, 0, 1]

vici_pts = array(pcd.points)[vici_idx]
all_pts = array(pcd.points)[idx]
# print(vici_pts)

mesh = get_mesh(now_pt, vici_pts)

# 变换后
print("Paint the 1500th point red.")
pick_idx = 1500
now_pt = array(pcd2.points[pick_idx])

pcd2.colors[pick_idx] = [1, 0, 0]  # 选一个点

print("Find its nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree2.search_knn_vector_3d(pcd2.points[pick_idx], vici_num)
vici_idx = idx[1:]
asarray(pcd2.colors)[vici_idx, :] = [0, 0, 1]

vici_pts = array(pcd2.points)[vici_idx]
all_pts = array(pcd2.points)[idx]

mesh2 = get_mesh(now_pt, vici_pts)

# print(dir(mesh))

diameter = np.linalg.norm(asarray(pcd2.get_max_bound()) - asarray(pcd2.get_min_bound()))

print("Define parameters used for hidden_point_removal")
camera = [150, 0, diameter]
radius = diameter * 1000

print("Get all points that are visible from given view point")
_, pt_map = pcd2.hidden_point_removal(camera, radius)

print("Visualize result")
pcd2 = pcd2.select_by_index(pt_map)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   pcd2,
                                   axis_pcd,
                                   mesh,
                                   # mesh2
                                   ],
                                  window_name='ANTenna3D',
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 2.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]
                                  # point_show_normal=True
                                  )
