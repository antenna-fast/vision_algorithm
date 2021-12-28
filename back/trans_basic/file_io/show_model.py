from numpy import *
import open3d as o3d

import os

from o3d_impl import get_non_manifold_vertex_mesh

# SHREC
# off文件测试数据读取

data_root = 'D:/SIA/Dataset/SHREC/SHREC/shrec_training/'
save_path = 'D:/SIA/Dataset/SHREC/SHREC/shrec_training_pcd/'

file_list = os.listdir(data_root)

for file_name in file_list:
    # f = open('D:/SIA/Dataset/SHREC/SHREC/shrec_training/0002.microholes.1.off')
    # f = open('D:/SIA/Dataset/SHREC/SHREC/shrec_training/0003.microholes.3.off')
    f = open('D:/SIA/Dataset/SHREC/SHREC/shrec_training/' + file_name)

    for i in range(2):
        a = f.readline()
        # print(a.split())
    pts_num = int(a.split()[0])
    tri_num = int(a.split()[1])

    pts_buf = []
    tri_buff = []

    for i in range(pts_num):
        a = f.readline()
        # print('a:', a)
        # print(array(list(map(float, list(a.split())))))
        # pt = array(list(map(float, list(a.split()))))
        pt = list(map(float, list(a.split())))
        pts_buf.append(pt)

    for i in range(tri_num):
        a = f.readline()
        tri = array(list(map(int, list(a.split())))) - 1
        tri_buff.append(tri)

    pts_buf = array(pts_buf)
    tri_buff = array(tri_buff)
    # print(pts_buf)

    mesh = get_non_manifold_vertex_mesh(pts_buf, tri_buff)
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts_buf)
    #
    # pcd.paint_uniform_color([0.0, 0.7, 0.1])

    save_name = os.path.splitext(file_name)[0] + '.ply'
    # o3d.io.write_point_cloud(save_path + save_name, mesh)
    o3d.io.write_triangle_mesh(save_path + save_name, mesh)

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

o3d.visualization.draw_geometries([mesh,
                                   # pcd2,
                                   axis_pcd,
                                   # mesh1,
                                   # mesh2
                                   ],
                                  window_name='ANTenna3D',
                                  # zoom=0.3412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[0.6172, 2.0475, 1.532],
                                  # up=[0, 0.1768, 0.9024]
                                  # up=[0, 0, 1]
                                  # point_show_normal=True
                                  )
