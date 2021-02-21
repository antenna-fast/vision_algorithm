import open3d as o3d
from dist import *  # 距离计算
import numpy as np
import os


# 加噪声函数
# 读入mesh和噪声参数，保存加了噪声的mesh
def add_noise_func(model_name, pcd, noise_rate, cov_rate):

    # bbox:
    # obb = pcd.get_oriented_bounding_box()
    # obb.color = (1, 0, 0)
    # aabb = array(obb.get_box_points())

    mesh_vertices = array(pcd.vertices)  # nx3
    print('pcd1_num:', len(mesh_vertices))

    # 噪声生成
    mean = array([0.0, 0, 0.0])
    cov = eye(3)*cov_rate
    pts_num = len(mesh_vertices)
    noise_pts_num = int(pts_num * noise_rate)
    noise = random.multivariate_normal(mean, cov, noise_pts_num)

    # 第二种 不改变点的整体数量,直接对采样的点添加
    rand_choose = np.random.randint(0, pts_num, noise_pts_num)
    mesh_vertices[rand_choose] += noise
    pcd.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    # 保存加了噪声的mesh
    save_root = 'D:/SIA/data_benchmark/mesh_add_noise/' + model_name + '/'
    if not(os.path.exists(save_root)):
        os.mkdir(save_root)

    save_path = save_root + str(noise_rate) + '.ply'
    o3d.io.write_triangle_mesh(save_path, pcd)


model_list = ['ant', 'armadillo', 'bird_3', 'bust', 'girl', 'hand_3', 'camel', 'teddy', 'table_2', 'rabbit']

# 噪声参数
noise_list = [0, 0.01, 0.03, 0.05, 0.07, 0.1]  # 在这个里面加上0

cov_rate = 0.00001

for model_name in model_list:
    # 读取mesh源文件
    mesh_dir = 'D:/SIA/data_benchmark/mesh/' + model_name + '.ply'

    pcd = o3d.io.read_triangle_mesh(mesh_dir)
    pcd.compute_vertex_normals()
    pcd.paint_uniform_color([0.0, 0.6, 0.1])

    for noise_rate in noise_list:
        add_noise_func(model_name, pcd, noise_rate, cov_rate)


if __name__ == '__main__':
    print()
