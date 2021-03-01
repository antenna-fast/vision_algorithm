from numpy import *

# 梯度

# 一维情况
# 错位相减
a = array([1, 2, 3, 4, 5, 7, 10, 6, 3])
data_len = len(a)
b = zeros(data_len)
b[:data_len-1] = a[1:]  # 错位
b[data_len-1] = a[data_len-1]  # 补上最后一位

delta = b-a
print(delta)


# 2D
