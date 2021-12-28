from o3d_impl import *

import os

# 从列表读取 画上去
# 在mesh文件上绘制圆球

model_root = 'D:/SIA/Dataset/SHREC/SHREC/shrec_training_pcd/'

# model_list = os.listdir(model_root)  # 应当是已经检测了的
# 马
model_list = ['0007.null.0.ply', '0007.localscale.1.ply', '0007.noise.4.ply', '0007.holes.5.ply',
              '0007.microholes.3.ply', '0007.topology.2.ply']
# 替换
for i in range(len(model_list)):
    model_list[i] = model_list[i].replace('0007', '0009')
print('model_list:', model_list)

# size = 0.5
size = 2.0
color = [0, 0, 1]

# 加载
for model_name in model_list:  # 所有的模型
    print('model_name:', model_name)
    data_path = model_root + model_name

    mesh = read_mesh(data_path)
    mesh.paint_uniform_color([0.7, 0.4, 0.2])
    # mesh.paint_uniform_color(array([65, 105, 225]) / 255)
    # mesh.paint_uniform_color(array([135, 206, 250]) / 255)

    pts_np = mesh2np(mesh)
    pcd = mesh2pcd(mesh)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 标注点文件
    label_path = '../measure/save_file_kpt_idx/SHREC11/' + model_name + '.txt'
    # idx_list = [65, 35, 74]
    idx_list = np.loadtxt(label_path).astype(int)

    s = keypoints_np_to_spheres(pts_np[idx_list], size=size, color=color)

    o3d.visualization.draw_geometries([
        mesh,
        s,  # 选中的点
    ],
        window_name='ANTenna3D',
        zoom=1,
        front=[0, 10, 0.01],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0.1],  # 用于确定相机右x轴
        point_show_normal=True
    )
