# 这是考虑到应付GT，这里要对关键点refine

# 方案：
# 1.DBSCAN  预计最佳
# 3.实验1 效果不是很好，于是使用gt的标注，进行过滤

# 本次使用DBSCAN

from numpy import *

from sklearn.cluster import DBSCAN

import open3d as o3d
import o3d_impl

import scipy.io as scio

# 读取索引，拿到待聚类点

# 不带b的
# model_list = ['antsPly', 'octopusPly']
# 带b的
# model_list = ['birdsPly', 'fishesPly', 'humansPly', 'spectaclesPly']
model_list = ['fishesPly']

# 加载
for model_name in model_list:  # 所有的模型
    print('model_name:', model_name)
    for model_num in range(1, 6):  # 部分变形  1-5
        print('model_num:', model_num)

        # 读取

file_name = '../save_file_kpt_idx/key_pts_buff_idx_' + model_name + '.txt'
key_pts = loadtxt(file_name, dtype='int')
# print(key_pts)

out_path = 'D:/SIA/data_benchmark/'
data_path = out_path + model_name + '.ply'
pcd = o3d.io.read_point_cloud(data_path)
# pcd = o3d.io.read_point_cloud('D:/SIA/科研/data/unzip_1/antsPly/1.ply')
print(pcd)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=10))
# pcd.paint_uniform_color([0.0, 0.5, 0.1])
pcd.paint_uniform_color([0.0, 0.6, 0.0])

# asarray(pcd.colors)[key_pts, :] = [1, 0, 0]

pts = array(pcd.points)[key_pts]

# print(pts)

# DBSCAN
# db = DBSCAN(eps=0.02, min_samples=6).fit(pts)  # 0.02  5  0.3多一点  ant
# db = DBSCAN(eps=0.04, min_samples=4).fit(pts)  # 0.02  5
db = DBSCAN(eps=0.04, min_samples=1).fit(pts)  # 0.02  5
core_samples_mask = zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# print(labels)  # 每个点的标签

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)  # 噪声点数  标签是-1
print('Estimated number of clusters: %d' % n_clusters_)  # 得到的是索引点中的索引，所以需
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = unique(labels)
# print(unique_labels)

refine_pts_buff = []

for ul in unique_labels:
    if ul == -1:
        continue  # 噪声

    c_mask = list(where(labels == ul)[0])  # 接下来对这些计算一下均值就ok了到那时不一定在上面了
    # print(c_mask)
    # for c in c_mask:
    #     # refine_pts_buff.append(c_mask)  # 二级索引，直接用这个去以及索引里面
    #     refine_pts_buff.append(c)  # 二级索引，直接用这个去以及索引里面
    refine_pts_buff.append(c_mask[0])

refine_buff = (array(refine_pts_buff))

final_idx = key_pts[refine_buff]

#
# 可视化
# print(refine_buff)

asarray(pcd.colors)[final_idx, :] = [1, 0, 0]

print('len:', len(final_idx))

# 存储新的索引
key_pts_buff_1 = array(final_idx) + 1  # matlab的索引从1开始
s = {'IP_vertex_indices': key_pts_buff_1}
save_path = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours/' + model_name + '.mat'
scio.savemat(save_path, s)

# for p in final_idx:
#     # pt = array(pcd.points)[p]
#     asarray(pcd.colors)[p] == [1, 0, 0]

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd,
                                   # pcd2,
                                   axis_pcd,
                                   ],
                                  window_name='ANTenna3D',
                                  )
