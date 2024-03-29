from o3d_impl import *


# 加载 mesh

mesh = read_mesh('D:/SIA/data_benchmark/mesh/Armadillo.ply')
mesh.paint_uniform_color([0.0, 0.5, 0.1])
# 构建搜索树
mesh_tree = o3d.geometry.KDTreeFlann(mesh)


# 用于GUI选点
def pick_points(pcd):
    print("  Press [shift + right click] to undo point picking")
    print("  After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

    return vis.get_picked_points()


picked_pt = pick_points(mesh)
print('pick:', picked_pt)  # 打印选定的点
