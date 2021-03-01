from numpy import *
import numpy as np
import random as rsample  # 产生随机数
import matplotlib.pyplot as plt

import time

from icp_using_svd import *

# RANSAC


# 直线
def get_kb(p1, p2):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return k, b


def get_pt2line_dist(pt, line_k, line_b):
    a, b, c = line_k, -1, line_b
    dist = np.abs(a * pt[0] + b * pt[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return dist


# 平面

# 圆
# 返回模型:至少三个点确定一个圆
# 参数:圆心 半径
def get_circle(pts):

    return 0

# 模型检验


# 位姿模型，已知点的对应关系。因此采样不是特别随机的
# 问题分析：至少几个点确定一个变换（系统模型）一般4个
# 模型检验：将变换应用到场景和目标，用度量标准度量

# 解决方案：去看书！


# sample_function  用于随机采样,得到不重复的n个索引
def sample_n(idx_list, n):
    res = rsample.sample(idx_list, n)
    return res


# 输入包含噪声的直线数据
# 输出直线参数, 内点, 外点
# 这里提供一个框架,换成其他的模型函数即可拟合其他的数据
# TODO: remove outliers, and expand it to other models

def ransac_line(pts, iter_num):
    # ransan 随机采样一致性算法，改进：记录变化趋势，减少随机性
    # 基本思想
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤,最后选出置信度最高的

    # pts: 所有数据点
    # iter_num: 迭代次数

    data_len = len(pts)
    idx_list = list(range(data_len))

    model_buff = []  # 保存k, b, dist

    while iter_num > 0:
        iter_num -= 1

        # 1.采样 得到随机不重合的点对
        random_idx = sample_n(idx_list, 2)
        pt1, pt2 = pts[random_idx[0]], pts[random_idx[1]]

        # 2.从以上样本估计模型
        k, b = get_kb(pt2, pt1)

        # 评价，将[x, y]代进去，如果点到直线的距离小于某个数值，该模型就得分
        dist = 0  # init dist
        for pt in pts:  # all point!  这里可以直接写成矩阵型形式
            dist_i = get_pt2line_dist(pt, k, b)  # 如果要拟合其他的模型,更换这个即可
            dist += dist_i  # 模型的总体性能
            # print('dist:', dist)
        # print('model score:', dist)
        model_buff.append([k, b, dist])  # 缓存模型以及评分

    model_buff = np.array(model_buff)  # iter_num x 3

    # finally we find min_dist from all model buffer
    idx = np.argmin(model_buff[:, 2])  # 取置信度最高的模型参数
    model_param = model_buff[idx]  # in [k, b]
    # print('model_param:', model_param)

    # 下面输出参数作为可选的(耗时).因为不一定所有任务都需要
    confidence = 0  # 模型置信度
    outlier_buff = []  # a set in {nx2}
    # 内外点阈值
    inlier_threshold = 0.5
    # 通过最佳模型去除外点
    for pt_idx in idx_list:  # all point!  这里可以直接写成矩阵型形式  Ax=b
        pt = pts[pt_idx]
        dist_i = get_pt2line_dist(pt, model_param[0], model_param[1])  # 如果要拟合其他的模型,更换这个即可

        if dist_i > inlier_threshold:  # 假设局外点更少，所以使用局外点减少次数
            outlier_buff.append(pt_idx)

    inlier_buff = list(set(idx_list) - set(outlier_buff))

    return model_param, confidence, inlier_buff, outlier_buff


# 输入匹配的点坐标
# 返回source在target中的位姿
# 目前直接使用的ICP，尚没有经过迭代和离群点剔除。如何应对模型上就有噪声？
# 迭代一次后，基本能找到离群点
def ransac_pose(source, target, iter_num, inlier_threshold):
    # ransan 随机采样一致性算法，改进：记录变化趋势，减少随机性
    # 基本思想
    # 1.从所有的数据中进行随机采样，得到一组足以建模的一组数据
    # 2.判断模型质量
    # 重复上述步骤,最后选出置信度最高的

    # source: 模型  注意 这是匹配好的
    # target: 场景
    # iter_num: 迭代次数
    # inlier_threshold: 度量距离小于阈值的视为内点

    data_len = len(source)
    idx_list = list(range(data_len))  # 创建索引随机采样的列表

    model_buff = []  # 保存k, b, dist
    score_buff = []  # 打分
    while iter_num > 0:
        iter_num -= 1

        # 1.采样 得到随机不重合的点对
        random_idx = sample_n(idx_list, 20)  # 每次随机采样几个
        # pt1, pt2 = pts[random_idx[0]], pts[random_idx[1]]
        sub_source = source[random_idx]  # 取出匹配的子集
        sub_target = target[random_idx]

        # 2.从以上样本估计模型 -- ICP  迭代几次看看  其实没啥效果：如果匹配正确，迭代一次就可以得到正确结果
        W = eye(len(sub_source))

        for _ in range(1):  # ICP迭代次数
            res_mat = icp_refine(sub_source, sub_target, W)  # 模型 场景 加权

            r_mat_res = res_mat[:3, :3]
            t_res = res_mat[:3, 3:].reshape(-1)

            # 评价，将全部点云代进去，看哪个位姿的误差小
            # 用位姿变换模型点云，然后与场景点云求距离
            # sub_source = dot(r_mat_res, (sub_source + t_res).T).T
            sub_source = dot(r_mat_res, sub_source.T).T + t_res  # 应当先旋转再平移

        source_trans = dot(r_mat_res, source.T).T + t_res  # 模型变换到场景上
        dist = var(norm(source_trans - target, axis=1))  # 计算全局欧氏距离  的方差？？方差：错误的一致性

        # print('model score:', dist)
        model_buff.append(res_mat)  # 缓存模型以及评分
        score_buff.append(dist)

    model_buff = np.array(model_buff)  # iter_num x 3
    score_buff = np.array(score_buff)

    # finally we find min_dist from all model buffer
    idx = np.argmin(score_buff)  # 取置信度最高的模型参数
    model_param = model_buff[idx]  # in [k, b]
    # print('model_param:', model_param)

    # 下面输出参数作为可选的(耗时).因为不一定所有任务都需要
    confidence = 0  # 模型置信度
    # 通过最佳模型去除外点   直接将

    source_trans = dot(r_mat_res, source.T).T + t_res  # 模型变换到场景上
    dist = np.linalg.norm(source_trans - target, axis=1)  # 计算全局欧氏距离  出来竟然是...一个数值了
    # print('dist:', dist)

    outlier_buff = np.where(dist > inlier_threshold)[0]  # 显示的内点 故阈值小外点多 内点少
    inlier_buff = array(list(set(idx_list) - set(outlier_buff))).astype(int)

    return model_param, confidence, inlier_buff, outlier_buff


if __name__ == '__main__':
    print('ransac to find a line')

    # 生成数据
    # define y=2*x + 3
    x = arange(0, 10, 0.1)
    y = 2 * x + 3 + np.random.rand(len(x))  # ok
    pts = array([x, y]).T

    # noise1 = np.random.rand(10, 2) * 3
    # noise1 = np.random.normal(10, 2, (20, 2))
    # noise2 = np.random.normal(5, 2, (20, 2))

    mean = array([10, 15])  # 正态分布
    cov = eye(2) * 3
    noise1 = np.random.multivariate_normal(mean, cov, 15)

    mean = array([-8, 15])  # 正态分布
    cov = eye(2) * 5
    noise2 = np.random.multivariate_normal(mean, cov, 25)
    # noise2 = np.random.uniform([0, 4], [10, 14], [10, 2])  # 均匀分布
    # noise2 = np.random.rand(10, 2) * 3 + [6, 20]

    pts = vstack((pts, noise1))
    pts = vstack((pts, noise2))

    print('pts_shape:', pts.shape)

    s_time = time.time()

    # data, iter_num
    iter_num = 30
    model_para, confi, inlier, outlier = ransac_line(pts, iter_num)

    e_time = time.time()
    print('time cost:{0}'.format(e_time - s_time))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ax.scatter(pts.T[0], pts.T[1])  # 原始数据
    ax.scatter(pts[inlier].T[0], pts[inlier].T[1], color='g')  # 区分内外点的
    ax.scatter(pts[outlier].T[0], pts[outlier].T[1], color='b')  # 区分内外点的

    # gen draw data
    x_test = array([-1, 10])
    y_test = model_para[0] * x_test + model_para[1]

    ax.plot(x_test, y_test, color='r')
    ax.plot(x_test, y_test, color='r')

    plt.show()
