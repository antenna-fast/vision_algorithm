import scipy.io as scio
from o3d_impl import *
import os

# source
data_root = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/MODEL_DATASET/'
# data_path = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/MODEL_DATASET/ant.mat'

file_list = os.listdir(data_root)

# out dir
out_path = 'D:/SIA/data_benchmark/'

for d_name in file_list:
    data_path = data_root + d_name

    data = scio.loadmat(data_path)
    # print(data['F'])
    # print(data['V'])

    all_pts = data['V']
    # tri_idx = data['F']

    # mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

    # 转换成ply，后面直接读取
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    out_name = d_name.replace('.mat', '.ply')
    o3d.io.write_point_cloud(out_path + out_name, pcd)

    # o3d.visualization.draw_geometries([
    #     pcd,
    #     # mesh,
    #     ],
    #     window_name='ANTenna3D',
    #     )

# save_data_name = 'idx.mat'
# scio.savemat(save_data_name, {'A': data['A']})

# o3d.visualization.draw_geometries([
#     pcd,
#     # mesh,
#     ],
#     window_name='ANTenna3D',
#     )


a = array([1, 2, 3, 4])
s = {'IP_vertex_indices': a}
scio.savemat('s.mat', s)
