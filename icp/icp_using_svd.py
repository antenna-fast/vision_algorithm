from numpy import *
from numpy.linalg import *
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

# SVD实现ICP
# 基本原理：最小二乘

# 定义变换
r = R.from_rotvec(pi / 180 * array([0, 20, 10]))  # 角度->弧度
# r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
r_mat = r.as_matrix()
t_vect = array([10, -2, -8], dtype='float')
print('r_mat:\n', r_mat)

# 定义被配准的点

pts_1 = []
for i in range(100):
    for j in range(100):
        pts_1.append([i, j, 0])

pts_1 = array(pts_1)

# 定义变换后的点
pts_2 = dot(r_mat, pts_1.T).T + t_vect
# print('pts_1:\n', pts_1)
# print('pts_2:\n', pts_2)

ax = plt.figure(1).gca(projection='3d')

ax.plot(pts_1.T[0], pts_1.T[1], pts_1.T[2], 'g.')
ax.plot(pts_2.T[0], pts_2.T[1], pts_2.T[2], 'r.')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

plt.title('point cloud')
plt.show()

# 每对点的权重
W = eye(len(pts_1))  # 假设每对点的权重都是1


# 求出两组点云的加权平均值 也就是加权重心
def get_sum(W, X):
    weighted_sum = sum(dot(W, X), axis=0)
    # weighted_sum = (dot(W, X))
    # w_vec_sum = sum(W.diagonal())
    w_vec_sum = sum(W)
    weighted_sum = weighted_sum / w_vec_sum
    return weighted_sum


pts_m_1 = get_sum(W, pts_1)
pts_m_2 = get_sum(W, pts_2)
# print('p_b:\n', pts_m_1)

# 定义xi yi
# 点云数据格式：nxd
Xi = pts_1 - pts_m_1
Yi = pts_2 - pts_m_2
# print('Xi:\n', Xi)
# print('Yi:\n', Yi)

# 构造S  Xi.T:dxn  W:nxn  Yi:nxd
S = dot(dot(Xi.T, W), Yi)
# print(S)

# 对S进行SVD分解
U, s, Vh = svd(S)
Vh = Vh.T

# 得到旋转矩阵R
mid_mat = eye(3)
mid_mat[2][2] = det(dot(Vh, U.T))

r_mat_res = dot(dot(Vh, mid_mat), U.T)

print('r_mat_res:\n', r_mat_res)

# 得到平移变换
# t = pts_m_1 - dot(inv(r_mat_res), pts_m_2.T)
t_res = pts_m_2 - dot(r_mat_res, pts_m_1.T)
print('t:\n', t_res)

# 变换后的点
pts_2_res = dot(inv(r_mat_res), (pts_2 - t_res).T).T  # - t_res
# pts_2_res = dot(r_mat_res, pts_1.T).T + t_res
# pts_2_res = pts_2_res.T
# print('pts_2_res:\n', pts_2_res)

ax = plt.figure(1).gca(projection='3d')

ax.plot(pts_1.T[0], pts_1.T[1], pts_1.T[2], 'g.')
ax.plot(pts_2_res.T[0], pts_2_res.T[1], pts_2_res.T[2], 'r.')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

plt.title('point cloud')
plt.show()

if __name__ == '__main__':
    print()
