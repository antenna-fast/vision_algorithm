from o3d_impl import *


# 在mesh文件上绘制圆球
# 暂时没有弄其他环上的,只是根据列表得到

# 加载
mesh = read_mesh('D:/SIA/data_benchmark/mesh/Armadillo.ply')
# pcd = pcd.voxel_down_sample(voxel_size=2)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))
mesh.paint_uniform_color([0.5, 0.5, 0.5])

pts_np = mesh2np(mesh)
pcd = mesh2pcd(mesh)

pcd_tree = o3d.geometry.KDTreeFlann(pcd)

idx_list = [6425, 355, 3174]

vici_num = 7


size = 0.01
color = [0, 0, 1]

s = keypoints_np_to_spheres(pts_np[idx_list], size=size, color=color)

o3d.visualization.draw_geometries([
    mesh,
    s,  # 选中的点
    # kpts_gt,
    # kpts_noise,
    # axis_pcd,
],
    window_name='ANTenna3D',
    zoom=1,
    front=[0, 10, 0.01],  # 相机位置
    lookat=[0, 0, 0],  # 对准的点
    up=[0, 1, 0],  # 用于确定相机右x轴
    point_show_normal=True
)