from numpy import *
from numpy.linalg import *
import open3d as o3d

from point_to_plan import pt_to_plan  # px, p, p_n  返回投影后的三维点
from 正交基变换 import *
from n_pt_plan import *
from dist import *  # 距离计算

from scipy.spatial import Delaunay


# 给定两个检测出来的关键点，在经过变换之后，看第二个检测出来的是不是在邻域范围之内
# 如果是，就认为是重合的

# 索引不变的情况
# 具体的思路：如果检测到，可以知道索引以及三维坐标
# 已知原坐标、现坐标、变换、邻域范围就可以
# 但是，这样只能是没有噪声的情况。因为加了噪声，索引和邻域就会变.如何解决

# 对于索引有改变的情况
# 执行思路:对每个关键点与场景中的点求最短距离 看最短距离是否小于阈值

# 加载 1
pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.6, 0.1])
pcd_np = array(pcd.points)

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

mean = array([1, 20, 1])
cov = eye(3)*1200
pts_num = len(pcd_trans)
noise_pts_num = int(pts_num * noise_rate)
noise = random.multivariate_normal(mean, cov, noise_pts_num)
# print('noise.shape:', noise.shape)

# 对噪声变换到场景坐标系

noise_trans = dot(r_mat, noise.T).T + t_vect

# 方式1 将噪声塞进去
pcd_trans = vstack((pcd_trans, noise_trans))

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.6, 0.1])
# pcd_trans_normal = array(pcd2.normals)

pcd2_np = array(pcd2.points)


# 索引一致的情况 (可以通过索引找到关键点)
# 给定两组点 以及变换
def get_repeate_rate(pcd_np, pcd_trans, r_mat, t_vect):
    # 首先将模型变换到场景
    pcd_np_trans = dot(r_mat, pcd_np.T).T + t_vect

    repeat_num = 0
    all_repeat = min(len(pcd_np), len(pcd_trans))  # 取出较小的一组

    for pt_idx in range(pts_num):  # 变换后的 参照点
        pt_1 = pcd_np_trans[pt_idx]
        pt_2 = pcd2_np[pt_idx]

        dist = sqrt(sum((pt_1 - pt_2) ** 2))
        # print(dist)

        if dist < 1:
            repeat_num += 1

    repeat_rate = repeat_num / all_repeat
    # print(repeat_rate)

    return repeat_rate


# 索引不一致
# 搜索得到最近邻

repeat_num = 0

vici_num = 2  # 2近邻 包含他自己

# 首先将模型变换到场景
pcd_np_trans = dot(r_mat, pcd_np.T).T + t_vect

pcd1_num = len(pcd_np)
pcd2_num = len(pcd2_np)

all_repeat = min(pcd1_num, pcd2_num)  # 取出较小的一组

for pt_idx in range(pts_num):
    pt_1 = pcd_np_trans[pt_idx]

    # print('len(pcd2_np):', len(pcd2_np))

    # 将变换后的点添加到变换后的场景中
    pcd2_np_temp = pcd2_np  # 每次都更新
    pcd2_np_temp = vstack((pcd2_np_temp, pt_1))  #
    # print('len(pcd2_np_temp):', len(pcd2_np_temp))

    # 找到最近邻点 看距离是否在范围内  其实这种方法也不需要提前知道变换
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_np_temp)

    # 构建搜索树
    pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)

    [k, idx, _] = pcd_tree_2.search_knn_vector_3d(pcd2.points[pcd2_num], vici_num)  # 最后一个点的近邻
    vici_idx = idx[1:]
    # asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]

    vici_pts = array(pcd2.points)[vici_idx]
    # all_pts = array(pcd.points)[idx]

    dist = sqrt(sum((pt_1 - vici_pts)**2))
    print(dist)

    if dist < 1:
        repeat_num += 1

    # print(pt_idx / all_repeat)

repeat_rate = repeat_num / all_repeat
print(repeat_rate)


if __name__ == '__main__':

    # 先直接使用原始数据
    # 基于索引,然后查找最近点,如果距离小于某个数值就算重合
    # for pt in pcd_np:

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
