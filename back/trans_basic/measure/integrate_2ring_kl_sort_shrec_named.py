from o3d_impl import *
from dist import *  # 距离计算

from detector import detect_2_ring_kl

import os

# 加载数据集里的点云，检测后保存顶点索引

data_root = 'D:/SIA/Dataset/SHREC/SHREC/shrec_training_pcd/'

# model_list = os.listdir(data_root)

master_list = [6]

# 马
model_list = ['0007.null.0.ply', '0007.localscale.1.ply',
              '0007.noise.4.ply', '0007.holes.5.ply',
              '0007.microholes.3.ply', '0007.topology.2.ply']
for i in range(len(model_list)):
        model_list[i] = model_list[i].replace('0007', '0009')  # 换成哪一个

# threshold = 0.5  # ant
# threshold = 1.9  # 数大 点少  camel
# threshold = 28.9  # 前面两个
# threshold = 1.3  # teddy
threshold = 2.3  # teddy

vici_num = 7
cut_num = vici_num - 5

for model_name in model_list:  # 所有的模型
        print('model_name:', model_name)
        data_path = 'D:/SIA/Dataset/SHREC/SHREC/shrec_training_pcd/' + model_name

        mesh = read_mesh(data_path)
        pcd = mesh2pcd(mesh)
        print(pcd)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
        pcd.paint_uniform_color([0.0, 0.6, 0.0])
        pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd)  # 构建搜索树

        pts_num = len(pcd.points)

        # 检测 得到索引
        # pcd_in, threshold, vici_num, cut_num
        key_pts_buff_1 = detect_2_ring_kl(pcd, threshold, vici_num, cut_num)  # 检测  返回索引

        print('key_pts_num:', len(key_pts_buff_1))
        savetxt('save_file_kpt_idx/SHREC11/' + model_name + '.txt', key_pts_buff_1, fmt='%d')

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
