from dist import *  # 距离计算
from o3d_impl import *

# TODO: 为了可视化

# 加载 1
pcd = o3d.io.read_point_cloud('../../data_ply/Armadillo.ply')
pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.5, 0.1])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

pcd_trans = array(pcd.points)  # nx3

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print('pcd1_num:', len(pcd.points))
# print('pcd2_num:', len(pcd2.points))

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_trans)

# print("Recompute the normal of the downsampled point cloud")
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd2.paint_uniform_color([0.0, 0.5, 0.1])
# pcd_trans_normal = array(pcd2.normals)

# 构建搜索树
pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)


# 用于GUI选点
def pick_points(pcd):
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


picked_pt = pick_points(pcd)
print('pick:', picked_pt)
