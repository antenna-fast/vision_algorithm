from o3d_impl import *
import os


# 实现mesh的下采样
# 1.Mesh decimation
# 2.Vertex clustering
# http://www.open3d.org/docs/release/tutorial/geometry/mesh.html

mesh_path = 'D:/SIA/data_benchmark/mesh/'
out_path = 'D:/SIA/data_benchmark/mesh_sample/'

color = [0.8, 0.5, 0]
file_list = os.listdir(mesh_path)

# 采样率
sample_list = []

for file_name in file_list:
    f_path = mesh_path + file_name
    mesh = read_mesh(f_path, mesh_color=color)

    # 采样

    # 保存
    save_path = out_path + file_name

    o3d.visualization.draw_geometries([
        # pcd,
        mesh,
        # pcd2,
        # axis_pcd,
    ],
        window_name='ANTenna3D',
        zoom=1,
        front=[0, 10, 0.01],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0],  # 用于确定相机右x轴
        point_show_normal=True
    )