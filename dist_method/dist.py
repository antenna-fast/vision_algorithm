from numpy import *
from numpy.linalg import *


# 欧式距离
def get_oclide(vec1, vec2):
    dist = sum((vec1 - vec2)**2)
    dist = sqrt(dist)
    return dist


# 曼哈顿距离
def get_manhattan(vec1, vec2):
    dist = sum(abs(vec1 - vec2))
    return dist


# 余弦相似度
def get_cosdist(vec1, vec2):
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return dist


# KL散度  非对称
def get_KL(vec1, vec2, vec_len):
    vec1 = (vec1+1)/2
    vec2 = (vec2+1)/2

    dist = 0
    for i in range(vec_len):
        dist += vec1[i] * log2(vec1[i] / vec2[i])
    # dist = sum()
    return dist


if __name__ == '__main__':
    print()
