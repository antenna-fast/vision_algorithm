from dist import *  # 距离计算
from o3d_impl import *

# 操作说明：
# 按住Shift，点击左键选中

# 加载 1
pcd = o3d.io.read_point_cloud('D:/SIA/data_benchmark/mesh/Armadillo.ply')
# pcd = pcd.voxel_down_sample(voxel_size=2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
pcd.paint_uniform_color([0.0, 0.5, 0.1])
# 构建搜索树
pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)

pcd_np = array(pcd.points)  # nx3
# print('pcd1_num:', len(pcd_np))


# 用于GUI 点云选点
def pick_points(pcd):
    print(" Press [shift + right click] to undo point picking")
    print(" After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print('')  # 间隔
    return vis.get_picked_points()


picked_pt = pick_points(pcd)
print('pick:', picked_pt)
