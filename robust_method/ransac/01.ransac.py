import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import time

# TODO: remove outliers, and expand it to other models

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


# 输入包含噪声的直线数据
# 输出直线参数, 内点, 外点
# 这里提供一个框架,换成其他的模型函数即可拟合其他的数据

def ransac_line(pts, sample_num):
    # ransan 随机采样一致性算法，改进：记录变化趋势，减少随机性
    # 基本思想
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤,最后选出置信度最高的

    data_len = len(pts)

    model_buff = []  # 保存k, b, dist
    # sample_num = 30  # over all sample time

    while sample_num > 0:
        sample_num -= 1
        # 1.采样 得到随机不重合的点对
        random_idx = np.random.randint(0, data_len, 2)  # only sample two points
        # 保证不一样的两个点
        while random_idx[0] == random_idx[1]:
            random_idx = np.random.randint(0, data_len, 2)
        # print('random_idx:', random_idx)
        pt1, pt2 = pts[random_idx[0]], pts[random_idx[1]]

        # 2.从以上样本估计模型
        k, b = get_kb(pt2, pt1)

        # 评价，将[x, y]代进去，如果点到直线的距离小于某个数值，该模型就得分
        dist = 0  # init dist
        for pt in pts:  # all point!
            dist_i = get_pt2line_dist(pt, k, b)  # 如果要拟合其他的模型,更换这个即可
            dist += dist_i  # 模型的总体性能
            # print('dist:', dist)
        # print('model score:', dist)

        model_buff.append([k, b, dist])  # 缓存模型以及评分

    model_buff = np.array(model_buff)  # sample_num x 3

    # finally we find min_dist from all model buffer
    idx = np.argmin(model_buff[:, 2])  # 取置信度最高的模型参数
    model_param = model_buff[idx]  # in [k, b]
    # print('model_param:', model_param)

    # 下面输出参数作为可选的(耗时).因为不一定所有任务都需要
    confidence = 0  # 模型置信度
    inlier = 0  # a set in {nx2}
    outlier = 0  # a set in {nx2}

    return model_param, confidence, inlier, outlier


if __name__ == '__main__':
    print('ransac to find a line')

    # 生成数据
    # define y=2*x + 3
    x = np.arange(0, 10, 0.1)
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

    mean = np.array([8, 15])  # 正态分布
    cov = np.eye(2)
    noise2 = np.random.multivariate_normal(mean, cov, 25)
    # noise2 = np.random.uniform([0, 4], [10, 14], [10, 2])  # 均匀分布
    # noise2 = np.random.rand(10, 2) * 3 + [6, 20]

    pts = np.vstack((pts, noise1))
    pts = np.vstack((pts, noise2))

    print('pts_shape:', pts.shape)

    s_time = time.time()

    # data, sample_num
    model_para, confi, inlier, outlier = ransac_line(pts, 30)

    e_time = time.time()
    print('time cost:{0}'.format(e_time - s_time))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pts.T[0], pts.T[1])  # 原始数据

    # gen draw data
    x_test = np.array([-1, 10])
    y_test = model_para[0] * x_test + model_para[1]
    ax.plot(x_test, y_test, color='r')
    plt.show()
