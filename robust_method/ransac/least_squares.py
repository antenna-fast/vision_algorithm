import numpy as np
import visdom
import matplotlib.pyplot as plt


# viz = visdom.Visdom(env='ransac')
# viz.scatter(np.random.random((1, 2)), win='line_detect')


def get_kb(p1, p2):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return k, b


def get_pt2line_dist(pt, line_k, line_b):
    a, b, c = line_k, -1, line_b
    dist = np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return dist


def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)


if __name__ == '__main__':
    print('ransac to find a line')

    # gen_data
    # define y=2*x + 3
    x = np.arange(0, 10, 0.2)
    y = 2 * x + 3 + np.random.rand(len(x))  # ok
    # print(x)
    # Y, X
    pts = np.array([x, y]).T

    # noise1 = np.random.rand(10, 2) * 3
    # noise1 = np.random.normal(10, 2, (20, 2))
    # noise2 = np.random.normal(5, 2, (20, 2))

    mean = np.array([2, 15])  # 正态分布
    cov = np.eye(2)
    noise1 = np.random.multivariate_normal(mean, cov, 25)

    mean = np.array([8, 12])  # 正态分布
    cov = np.eye(2)
    noise2 = np.random.multivariate_normal(mean, cov, 25)
    # noise2 = np.random.uniform([0, 4], [10, 14], [10, 2])  # 均匀分布
    # noise2 = np.random.rand(10, 2) * 3 + [6, 20]

    pts = np.vstack((pts, noise1))
    pts = np.vstack((pts, noise2))
    print('pts_shape:', pts.shape)

    data_len = len(pts)

    a0, a1 = linear_regression(pts.T[0], pts.T[1])

    fig = plt.figure()
    # print(dir(fig))
    ax = fig.add_subplot(111)
    ax.scatter(pts.T[0], pts.T[1], color='g')

    # gen draw data
    x_test = np.array([-1, 10])
    y_test = a1 * x_test + a0
    ax.plot(x_test, y_test, color='r')
    plt.show()
