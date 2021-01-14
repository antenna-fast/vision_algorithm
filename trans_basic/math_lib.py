from numpy import *


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


a = array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 1, 1]])

var_a = get_var(a)

# var_a = var(a, axis=0)

# print(mean_a)
# print(var_a)


a = array([1, 1, 2])
print(var(a, axis=0))
