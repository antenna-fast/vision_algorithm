from o3d_impl import *
import os


# 实现mesh的下采样
# 1.Mesh decimation
# 2.Vertex clustering
# http://www.open3d.org/docs/release/tutorial/geometry/mesh.html

# 采样完的只进行可视化,看是不是一致

mesh_path = 'D:/SIA/data_benchmark/mesh/'  # source
out_path = 'D:/SIA/data_benchmark/mesh_sample/'  # target

color = [0.8, 0.5, 0]
file_list = os.listdir(mesh_path)

# 采样率
sample_list = [32, 48, 64, 96]

for sample_num in sample_list:
    for file_name in file_list:
        model_name = file_name.replace('.ply', '')
        f_path = mesh_path + file_name  # 读取路径
        mesh = read_mesh(f_path, mesh_color=color)

        # 采样  几种策略/定量描述:使用顶点数和面片数
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / sample_num
        # print(f'voxel_size = {voxel_size:e}')

        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)

        print(f'Simplified mesh has {len(mesh_smp.vertices)} '
              f'vertices and {len(mesh_smp.triangles)} triangles')

        # 保存
        save_path = out_path + model_name + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_file_path = save_path + str(sample_num) + '_' + file_name
        o3d.io.write_triangle_mesh(save_file_path, mesh_smp)

        if 0:
            o3d.visualization.draw_geometries([
                # pcd,
                # mesh,
                mesh_smp,
                # axis_pcd,
            ],
                window_name='ANTenna3D',
                zoom=1,
                front=[0, 10, 0.01],  # 相机位置
                lookat=[0, 0, 0],  # 对准的点
                up=[0, 1, 0],  # 用于确定相机右x轴
                # point_show_normal=True
            )