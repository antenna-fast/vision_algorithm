from numpy import *
from numpy.linalg import *
import open3d as o3d

from point_to_plan import pt_to_plan  # px, p, p_n  返回投影后的三维点
from 正交基变换 import *
from n_pt_plan import *
from dist import *  # 距离计算

from scipy.spatial import Delaunay


# 加载 1
pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.6, 0.1])

# 加载 2
pcd_trans = array(pcd.points)  # nx3

# 定义变换
r = R.from_rotvec(pi / 180 * array([30, 30, 30]))  # 角度->弧度
# r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([150, -2, -8], dtype='float')
print('r_mat:\n', r_mat)

pcd_trans = dot(r_mat, pcd_trans.T).T
pcd_trans = pcd_trans + t_vect

# 加噪声
noise_rate = 0.1  # 噪声占比

# mean = array([1, 20, 1])
mean = array([1, 0, 1])
# cov = eye(3)*1000
cov = eye(3)*10
pts_num = len(pcd_trans)
noise_pts_num = int(pts_num * noise_rate)
noise = random.multivariate_normal(mean, cov, noise_pts_num)
# print('noise.shape:', noise.shape)

# 对噪声变换到场景坐标系

noise_trans = dot(r_mat, noise.T).T + t_vect

# 方式1 将噪声塞进去
# pcd_trans = vstack((pcd_trans, noise_trans))


# 第二种 不改变点的整体数量,直接对采样的点添加
rand_choose = np.random.randint(0, pts_num, noise_pts_num)
pcd_trans[rand_choose] += noise

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.6, 0.1])
# pcd_trans_normal = array(pcd2.normals)


print('pcd1_num:', len(pcd.points))
print('pcd2_num:', len(pcd2.points))

if __name__ == '__main__':
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

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
