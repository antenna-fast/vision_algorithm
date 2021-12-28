from o3d_impl import *
from dist import *  # 距离计算

from detector import detect_2_ring_kl

# 加载数据集里的点云，检测后保存顶点索引
# 在matlab里面进行可视化（有mesh）

# 不带b的
# model_list = ['antsPly', 'octopusPly']
# 带b的
# model_list = ['birdsPly', 'fishesPly', 'humansPly', 'birdsPly', 'spectaclesPly']
# model_list = ['fishesPly']
# model_list = ['teddyPly']
# model_list = ['pliersPly']  # 钳子
# model_list = ['dinosaursPly']  #
model_list = ['chairsPly']  #

# threshold = 0.5  # ant
# threshold = 1.9  # 数大 点少  camel
# threshold = 28.9  # 前面两个
threshold = 20.9  # teddy

vici_num = 7
cut_num = vici_num - 4

for model_name in model_list:  # 所有的模型
    print('model_name:', model_name)
    for model_num in range(1, 6):  # 部分变形  1-5
        print('model_num:', model_num)

        # 不带b
        # data_path = 'D:/SIA/Dataset/SHREC/unzip_1/' + model_name + '/' + str(model_num) + '.ply'
        data_path = 'D:/SIA/Dataset/SHREC/unzip_1/' + model_name + '/b' + str(model_num) + '.ply'

        pcd = o3d.io.read_point_cloud(data_path)
        print(pcd)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
        pcd.paint_uniform_color([0.0, 0.6, 0.0])
        pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)  # 构建搜索树

        pts_num = len(pcd.points)

        # 检测 得到索引
        # pcd_in, threshold, vici_num, cut_num
        key_pts_buff_1 = detect_2_ring_kl(pcd, threshold, vici_num, cut_num)  # 检测  返回索引

        print('key_pts_num:', len(key_pts_buff_1))

        savetxt('save_file_kpt_idx/SHREC/' + model_name + '_' + str(model_num) + '.txt', key_pts_buff_1, fmt='%d')
        # savetxt('save_file_kpt_idx/key_pts_buff_idx_' + model_name + '.txt', key_pts_buff_1, fmt='%d')

        # matlab服务  此时用不到
        # key_pts_buff_1 = array(key_pts_buff_1) + 1  # matlab的索引从1开始
        # s = {'IP_vertex_indices': key_pts_buff_1}
        # save_path = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours' \
        #             '/SHREC/' + model_name + '/' + str(model_num) + '.mat '
        # io.savemat(save_path, s)

# 可视化
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#
# o3d.visualization.draw_geometries([pcd,
#                                    axis_pcd,
#                                    ],
#                                   window_name='ANTenna3D',
#                                   # zoom=0.3412,
#                                   # front=[0.4257, -0.2125, -0.8795],
#                                   # lookat=[2.6172, 2.0475, 1.532],
#                                   # up=[-0.0694, -0.9768, 0.2024]
#                                   # point_show_normal=True
#                                   )
