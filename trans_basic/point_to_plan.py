from numpy import *
import matplotlib.pyplot as plt

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
a = array([-5, 5, 5]).astype(int)


# 输入:点 平面 平面法向量
# 输出:投影点
def pt_to_plan(a, p, p_n):
    # 参数t
    t = -1 * (p[0] * a[0] + p[1] * a[1] + p[2] * a[2] + p[3]) / (p[0] * p_n[0] + p[1] * p_n[1] + p[2] * p_n[2])
    # print(t)

    x = a + t * p_n

    return x


if __name__ == '__main__':
    x = pt_to_plan(a, p, p_n)

    a = a[:, newaxis]
    x = x[:, newaxis]

    ax = plt.figure(1).gca(projection='3d')

    ax.plot(pts_buff.T[0], pts_buff.T[1], pts_buff.T[2], 'g.')
    ax.plot(a[0], a[1], a[2], 'o')
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

