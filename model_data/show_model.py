from numpy import *
import open3d as o3d
from o3d_impl import *


# 测试数据读取

# f_path = '../../SHREC/shrec_training/0002.all.1.off'
# f_path = '../../SHREC/shrec_training/0002.all.2.off'
# f_path = '../../SHREC/shrec_training/0002.holes.1.off'
f_path = '../../SHREC/shrec_training/0002.null.0.off'
f = open(f_path)
# print(f.readlines())

# mesh = o3d.io.read_triangle_mesh(f_path)

for i in range(2):
    a = f.readline()
    # print(a.split())
# 顶点
pts_num = int(a.split()[0])
# 面片
tri_num = int(a.split()[1])

pts_buf = []
for i in range(pts_num):
    a = f.readline()
    # print('a:', a)
    # print(array(list(map(float, list(a.split())))))
    # pt = array(list(map(float, list(a.split()))))
    pt = list(map(float, list(a.split())))
    pts_buf.append(pt)

pts_buf = array(pts_buf)

# tri_buf = []
# for i in range(tri_num):
#     a = f.readline()
#     # idx = list(map(float, list(a.split())))
#     idx = list(map(int, list(a.split())))
#     tri_buf.append(idx)
#     print(idx)
#
# tri_buf = array(tri_buf)
# mesh = get_non_manifold_vertex_mesh(pts_buf, tri_buf)  # 不行的原因：本来就存在孔洞

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_buf)

pcd.paint_uniform_color([0.7, 0.7, 0.0])


axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   # pcd2,
                                   axis_pcd,
                                   # mesh,
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
