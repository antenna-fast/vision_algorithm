from o3d_impl import *

# 加载模型

noise_rate = 0
# noise_rate = 0.03
# noise_rate = 0.05
# noise_rate = 0.07
# noise_rate = 0.1

# 上层：不同模型
model_list = ['ant', 'armadillo', 'bird_3', 'bust', 'girl', 'hand_3', 'camel', 'teddy', 'table_2', 'rabbit']
# for model_name in model_list:
# model_name = 'armadillo'  # ok
# model_name = 'ant'
# model_name = 'camel'  # ok
# model_name = 'teddy'  # ok
# model_name = 'rabbit'  # ok
model_name = 'girl'

# 每个文件：不同vici_num的文件
vici_num_list = [5, 6, 7, 8, 9, 10, 11]

data_root = 'D:/SIA/data_benchmark/'

for vici_num in vici_num_list:  # 关键点索引
    print('vici_num:', vici_num)
    # 加载保存的 1.对应的模型 2.关键点索引
    mesh_gt_dir = data_root + 'mesh_add_noise/' + model_name + '/' + '0.ply'  # noise == 0
    mesh_noise_dir = data_root + 'mesh_add_noise/' + model_name + '/' + str(noise_rate) + '.ply'  # others

    mesh_gt = read_mesh(mesh_gt_dir)
    mesh_noise = read_mesh(mesh_noise_dir)

    # print('idx_gt:', idx_gt_path)
    # print('mesh_gt_dir:', mesh_gt_dir)
    # print('mesh_gt:', mesh_gt)

    all_pts_gt = mesh2np(mesh_gt)
    all_pts_noise = mesh2np(mesh_noise)

    # 加载关键点索引
    idx_gt_path = data_root + 'mesh_add_noise_save/' + model_name + '/' + str(vici_num) + '_' + str(0) + '.txt'
    idx_noise_path = data_root + 'mesh_add_noise_save/' + model_name + '/' + str(vici_num) + '_' + str(
        noise_rate) + '.txt'

    idx_gt = loadtxt(idx_gt_path).astype('int')
    idx_noise = loadtxt(idx_noise_path).astype('int')
    # print('idx_gt:', idx_gt)

    # 将关键点idx paint到mesh上面
    size = 0.005
    kpts_gt = keypoints_np_to_spheres(all_pts_gt[idx_gt], size=size)
    kpts_noise = keypoints_np_to_spheres(all_pts_noise[idx_noise], size=size)

    # 可视化
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([
        # mesh_gt,
        mesh_noise,
        # kpts_gt,
        kpts_noise,
        # axis_pcd,
    ],
        window_name='ANTenna3D',
        zoom=1,
        front=[0, 10, 0.01],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0],  # 用于确定相机右x轴
        point_show_normal=True
    )
