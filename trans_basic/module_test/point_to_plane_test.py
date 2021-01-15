from numpy import *
import matplotlib.pyplot as plt

from point_project import *


p = array([1, 1, 1, 1])
p_n = p[:3]

# 注意 平面要按照方程生成！
pts_buff = []
for i in range(-10, 10):
    for j in range(-10, 20):
        z = -1 * (p[3] + p[0] * i + p[1] * j) / p[2]
        pts_buff.append([i, j, z])

pts_buff = array(pts_buff)

# 被投影点
a = array([[-5, 9, 5],
           [1, 10, 8],
           [-10, -10, -2]
           ]).astype(int)


if __name__ == '__main__':
    x = pt_to_plane(a, p, p_n)

    # a = a[:, newaxis]
    # x = x[:, newaxis]

    ax = plt.figure(1).gca(projection='3d')

    ax.plot(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g.')
    # a = a.squeeze()
    print(a)
    a = a.T
    ax.plot(a[0], a[1], a[2], 'o')
    # print(x)
    x = x.T
    # x = x.squeeze()
    ax.plot(x[0], x[1], x[2], 'r.')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # plt.axis("equal")
    # plt.axis("auto")
    # ax.set_aspect('equal')
    ax.set_zlim(-10, 10)
    plt.title('point cloud')
    plt.show()

