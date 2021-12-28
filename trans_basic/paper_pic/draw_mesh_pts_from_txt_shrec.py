from o3d_impl import *

# 从列表读取 画上去
# 在mesh文件上绘制圆球

# 不带b的
# model_list = ['antsPly', 'octopusPly']
# 带b的
# model_list = ['birdsPly', 'fishesPly', 'humansPly', 'spectaclesPly']
# model_list = ['fishesPly', 'humansPly', 'spectaclesPly']
# model_list = ['humansPly', 'spectaclesPly']
# model_list = ['spectaclesPly']  # 太阳镜
# model_list = ['fishesPly']  #
# model_list = ['teddyPly']  # 熊
# model_list = ['pliersPly']  # 钳子
# model_list = ['dinosaursPly']  #
model_list = ['chairsPly']  #

# size = 0.5
size = 1.0
color = [0, 0, 1]

# 加载
for model_name in model_list:  # 所有的模型
    print('model_name:', model_name)
    for model_num in range(1, 6):  # 部分变形  1-5
        print('model_num:', model_num)

        # 不带b
        # data_path = 'D:/SIA/Dataset/SHREC/unzip_1/' + model_name + '/' + str(model_num) + '.ply'
        data_path = 'D:/SIA/Dataset/SHREC/unzip_1/' + model_name + '/b' + str(model_num) + '.ply'

        mesh = read_mesh(data_path)
        mesh.paint_uniform_color([0.6, 0.5, 0.5])

        pts_np = mesh2np(mesh)
        pcd = mesh2pcd(mesh)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # 标注点文件
        label_path = '../measure/save_file_kpt_idx/SHREC/' + model_name + '_' + str(model_num) + '.txt'
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
