import open3d as o3d

from base_trans import *
from dist import *  # 距离计算

# 现在可以直接可视化了！

# 加载 1

mesh_dir = 'D:/SIA/data_benchmark/mesh/armadillo.ply'

pcd = o3d.io.read_triangle_mesh(mesh_dir)
# pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.compute_vertex_normals()
pcd.paint_uniform_color([0.0, 0.6, 0.1])

# print(array(pcd.vertices))


# 加载 2
mesh_vertices = array(pcd.vertices)  # nx3
#
# # 定义变换
r = R.from_rotvec(pi / 180 * array([30, 30, 30]))  # 角度->弧度
# # r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([0, 0, -1], dtype='float')
# print('r_mat:\n', r_mat)
#
# pcd_trans = dot(r_mat, pcd_trans.T).T
# pcd_trans = pcd_trans + t_vect

# 加噪声
noise_rate = 0.1  # 噪声占比
# print(dir(pcd))

# bbox:
obb = pcd.get_oriented_bounding_box()
obb.color = (1, 0, 0)
aabb = array(obb.get_box_points())
# print(aabb)
# 还可以再加颜色

# mean = array([1, 20, 1])
mean = array([0.0, 0, 0.0])
# cov = eye(3)*1000
cov = eye(3)*0.00001
pts_num = len(mesh_vertices)
noise_pts_num = int(pts_num * noise_rate)
noise = random.multivariate_normal(mean, cov, noise_pts_num)
# print('noise.shape:', noise.shape)

# 对噪声变换到场景坐标系
noise_trans = dot(r_mat, noise.T).T + t_vect

# 第二种 不改变点的整体数量,直接对采样的点添加
rand_choose = np.random.randint(0, pts_num, noise_pts_num)
mesh_vertices[rand_choose] += noise
pcd.vertices = o3d.utility.Vector3dVector(mesh_vertices)

mesh_vertices[rand_choose] += noise
# pcd2 = o3d.geometry.TriangleMesh()
pcd.vertices = o3d.utility.Vector3dVector(mesh_vertices)

# print("Recompute the normal of the downsampled point cloud")
# pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
# pcd2.paint_uniform_color([0.0, 0.6, 0.1])
# pcd_trans_normal = array(pcd2.normals)


print('pcd1_num:', len(pcd.vertices))
# print('pcd2_num:', len(pcd2.vertices))

if __name__ == '__main__':
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd,
                                       # pcd2,
                                       obb,
                                       # axis_pcd,
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
