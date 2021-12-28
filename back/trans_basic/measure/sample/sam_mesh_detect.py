from o3d_impl import *
from detector import *
from dist import *

# 对不同的模型采样率检测,完直接可视化

# model_name = 'ant'
model_name = 'camel'
# model_name = 'ant'

# sample_list = [32, 48, 64, 96]
# sample_num = 32
# sample_num = 48
# sample_num = 64
sample_num = 96

data_root = 'D:/SIA/data_benchmark/mesh_sample/'

# 加载数据
mesh_path = data_root + model_name + '/' + str(sample_num) + '_' + model_name + '.ply'

color = [0.7, 0.4, 0]
mesh = read_mesh(mesh_path, mesh_color=color)

pcd = mesh2pcd(mesh)
# 检测
# thres = 3  # 32
thres = 3  # 64
vici_num = 7
key_pts_idx = detect_2_ring_kl(pcd, threshold=thres, vici_num=vici_num, cut_num=vici_num-3)
print('关键点数量:', len(key_pts_idx))

# 关键点转球
size = 0.008
color = [0, 1, 1]
pcd_np = mesh2np(mesh)
s = keypoints_np_to_spheres(pcd_np[key_pts_idx], size=size, color=color)

# 可视化
o3d.visualization.draw_geometries([
    # pcd,
    mesh,
    s,
    # mesh_smp,
    # axis_pcd,
],
    window_name='ANTenna3D',
    zoom=1,
    front=[0, 10, 0.01],  # 相机位置
    lookat=[0, 0, 0],  # 对准的点
    up=[0, 1, 0],  # 用于确定相机右x轴
    # point_show_normal=True
)