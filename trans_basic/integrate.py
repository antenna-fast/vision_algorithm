from numpy import *
from numpy.linalg import *
import open3d as o3d

from point_to_plan import pt_to_plan  # px, p, p_n  返回投影后的三维点
from 正交基变换 import *
from n_pt_plan import *

from scipy.spatial import Delaunay

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

# 找到邻域
# 输入当前点 邻域
# 输出
def get_mesh(now_pt, vici_pts):
    # 对每个邻域:
    # 输入当前点以及邻域，得到邻域局部坐标系 得到法向量等等
    coord = get_coord(now_pt, vici_pts)  # 返回:列向量表示三个轴
    normal = coord[:, 2]  # 第三列
    # print('coord:\n', coord)

    # 还有一步 利用法向量和中心点,得到平面方程[ABCD]
    p = get_plan(normal, now_pt)

    # * 找到拓扑结构 START

    # 将周围的点投影到平面
    plan_pts = []
    for pt in all_pts:  # 投影函数可以升级  向量化
        pt_temp = pt_to_plan(pt, p, normal)  # px p pn
        plan_pts.append(pt_temp)

    plan_pts = array(plan_pts)

    # 将投影后的点旋转至z轴,得到投影后的二维点
    coord_inv = inv(coord)  # 反变换
    # rota_pts = dot(coord_inv, all_pts.T).T  # 将平面旋转到与z平行
    rota_pts = dot(coord_inv, plan_pts.T).T  # 将平面旋转到与z平行

    # rota_pts[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动
    pts_2d = rota_pts[:, 0:2]

    # Delauney三角化
    tri = Delaunay(pts_2d)
    tri_idx = tri.simplices  # 三角形索引
    # print(tri_idx)

    # 可视化二维的投影
    plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
    plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
    plt.show()

    # 根据顶点和三角形索引创建mesh
    mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

    # * 找到拓扑结构 END

    # 求mesh normal
    mesh.compute_triangle_normals()
    print('triangle_normals:\n', array(mesh.triangle_normals))

    return mesh


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
