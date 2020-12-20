import numpy as np
import visdom
import matplotlib.pyplot as plt

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
    model_buff = []  # 保存k, b, dist
    sample_num = 1000  # over all sample time

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
        for pt in pts:  # all point!
            dist += get_pt2line_dist(pt, k, b)
            # print('dist:', dist)

        print('model score:', dist)

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

    # viz.line([[0, 0], [10, 10]],
    #          # x,
    #          win='line_detect',
    #          update='replace',
    #          opts=dict(title='line demo', caption='random.'))

    fig = plt.figure()
    # print(dir(fig))
    ax = fig.add_subplot(111)
    ax.scatter(pts.T[0], pts.T[1])

    # gen draw data
    x_test = np.array([-1, 10])
    y_test = model_buff[idx][0] * x_test + model_buff[idx][1]
    ax.plot(x_test, y_test, color='r')
    plt.show()
