from numpy import *
from numpy.linalg import *


# 欧式距离
def get_Euclid(vec1, vec2):
    dist = sum((vec1 - vec2)**2)
    dist = sqrt(dist)
    return dist


# 曼哈顿距离
def get_Manhattan(vec1, vec2):
    dist = sum(abs(vec1 - vec2))
    return dist


# 余弦相似度
def get_cos_dist(vec1, vec2):
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))  # 有时候vec norm=0...
    return dist


# KL散度  非对称
def get_KL(vec1, vec2, vec_len):
    dist = 0
    for i in range(vec_len):  # 如果有等于0的
        dist += vec1[i] * log2(vec1[i] / vec2[i])
    # dist = sum()
    return dist


# # test kl  非负：P>Q即可
# a = array([0.1, 0.1, 0.3, 0.4])
# b = array([0.1, 0.9, 0.7, 0.6])
# a = a/sum(a)
# b = b/sum(b)
# print('kl:', get_KL(a, b, 4))


# 输入一个序列  输出是否不平衡
def get_unbalance(vec, threshold):
    len_vec = len(vec)
    sort_vec = sort(vec)
    sort_vec_cut = sort_vec[:len_vec-1]  # 前面的len-1个

    diff = sort_vec[1:] - sort_vec_cut  # 一个以后的-前面n-1个  包含n-1个元素  为了快速计算

    # print('sort_vec:', sort_vec)
    # print('diff:', diff)
    res = 0  # 是否平衡点
    if max(diff) > threshold:
        res = 1

    return res


if __name__ == '__main__':

    # 不平衡点测试
    a = array([3, 2, 4, 5, 1, 9])  # 序列
    res = get_unbalance(a, 1)
    print('res:', res)
