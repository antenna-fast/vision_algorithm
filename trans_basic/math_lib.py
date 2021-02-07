from numpy import *
from numpy.linalg import *


# 一维
def get_var(a):
    # 求平均
    mean_a = mean(a, axis=0)  #
    # print(a - mean_a)

    # 做差 求平方 求平均
    diff = a - mean_a

    diff_2 = sum(diff ** 2, axis=0)
    # print('diff_2:\n', diff_2)

    var_a = diff_2 / len(a)
    # print(var_a)

    return var_a


def vec_proj_scale(vec_1, vec_2):
    # 将向量1投影到向量2  标量
    norm_vec_1 = norm(vec_1)
    cos = dot(vec_1, vec_2) / (norm_vec_1 * norm(vec_2))
    res = norm_vec_1 * cos
    return res


def vec_proj_vec(vec_1, vec_2):
    # 向量投影
    norm_vec_2 = norm(vec_2)
    res = dot(dot(vec_1, vec_2), vec_2) / (norm_vec_2**2)
    return res


if __name__ == '__main__':
    a = array([3, 3])
    b = array([1, 1])
    # res = vec_proj_scale(a, b)
    res = vec_proj_vec(a, b)
    print(res)
