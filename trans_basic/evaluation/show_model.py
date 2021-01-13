from numpy import *
import open3d as o3d

# 测试数据读取

f = open('shrec_training/0002.holes.1.off')
# print(f.readlines())

for i in range(2):
    a = f.readline()
    # print(a.split())
pts_num = int(a.split()[0])

pts_buf = []

for i in range(pts_num):
    a = f.readline()
    # print('a:', a)
    # print(array(list(map(float, list(a.split())))))
    # pt = array(list(map(float, list(a.split()))))
    pt = list(map(float, list(a.split())))
    pts_buf.append(pt)

pts_buf = array(pts_buf)

# print(pts_buf)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_buf)

pcd.paint_uniform_color([0.0, 0.7, 0.1])

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
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
