import numpy as np
from numpy import *
import visdom
import matplotlib.pyplot as plt

import time

# TODO: remove outliers, and expand it to other models


def get_kb(p1, p2):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return k, b


def get_pt2line_dist(pt, line_k, line_b):
    a, b, c = line_k, -1, line_b
    dist = np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return dist


if __name__ == '__main__':
    print('ransac to find a line')

    # gen_data
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

    # viz.scatter(pts,
    #             # x,
    #             win='line_detect',
    #             update='replace',
    #             opts=dict(title='line demo', caption='random.'))

    # ransan 随机采样一致性算法，改进：记录变化趋势，减少随机性
    # 基本思想
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤

    data_len = len(pts)

    # 每个点的权重
    data_w = ones(data_len).astype(int)  # 初始化权重  这个要进行更新  目的是驱动函数找到内点 内点权重要变大

    model_buff = []  # 保存k, b, dist
    sample_num = 30  # over all sample time

    s_time = time.time()

    resample_rate = 1

    while sample_num > 0:
        sample_num -= 1
        # 1.采样
        random_idx = np.random.randint(0, data_len, 2)  # only sample two points
        # 保证不一样的两个点
        while random_idx[0] == random_idx[1]:
            random_idx = np.random.randint(0, data_len, 2)
        # print('random_idx:', random_idx)

        # 2.从以上样本估计模型
        pt1, pt2 = pts[random_idx[0]], pts[random_idx[1]]
        k, b = get_kb(pt2, pt1)
        # print('{0}, {1}'.format(k, b))

        # 评价，将[x, y]代进去，如果点到直线的距离小于某个数值，该模型就得分
        dist = 0  # init dist

        resample_num = int(resample_rate * data_len)
        # print('data_w:', data_w)
        resample_list = argsort(data_w)[:resample_num]  # 从小到大
        # resample_list = argsort(data_w)
        # print('sample_num:', resample_num)

        # pts_sample = pts[]  # 对模型点再次进行采样  里面对内点赋予权重
        # for pt in pts:  # sampled point!
        for i in resample_list:  # sampled point!
            pt = pts[i]
            dist_i = get_pt2line_dist(pt, k, b)

            dist += dist_i  # 模型的总体性能
            # print('dist:', dist)

            if sample_num > 10:  # 采样一定程度之后 此时置信度边高
                data_w[i] = dist_i  # 越小越好
                resample_rate -= 0.0005  # 不能一直减!
                # resample_rate = 0.6

        # print('model score:', dist)

        model_buff.append([k, b, dist])

        # to visualization
        # # finally we find min_dist from all model buffer
        # model_buff = np.array(model_buff)
        # print(model_buff.shape)
        #
        # idx = np.argmin(model_buff[:, 2])
        # print(model_buff[idx])

    # finally we find min_dist from all model buffer
    model_buff = np.array(model_buff)
    print(model_buff.shape)

    idx = np.argmin(model_buff[:, 2])
    print(model_buff[idx])

    e_time = time.time()

    print('time cost:{0}'.format(e_time - s_time))

    fig = plt.figure()
    # print(dir(fig))
    ax = fig.add_subplot(111)
    ax.scatter(pts.T[0], pts.T[1])

    # gen draw data
    x_test = np.array([-1, 10])
    y_test = model_buff[idx][0] * x_test + model_buff[idx][1]
    ax.plot(x_test, y_test, color='r')
    plt.show()
