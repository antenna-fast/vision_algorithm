"""
* 非极大值抑制

3邻域内，如果是最大的就输出索引 且跳转到i+2
如果不是,则继续向前爬动,直到当前的大于下一个（且不超出列表长度）
"""

import numpy as np
import matplotlib.pyplot as plt


a = [1, 2, 2, 3, 2, 1, 1, 4, 5, 6.2, 6.1, 3, 1, 2, 5]

len_a = len(a)

i = 0  # 开始点
MaxIdx = []

while i < len_a - 1:  # i作为索引
    if a[i] > a[i + 1]:
        if a[i] >= a[i - 1]:
            MaxIdx.append(i)

    else:  # a[i] < a[i+1]
        i += 1  # 先跳一个格试探一下 不加也没关系
        while i < len_a - 1 and a[i] <= a[i + 1]:  # 向前爬  条件位置不能互换
            i += 1

        if i < len_a:  # 这时候 a[i] > a[i+1] &&  a[i] < a[i+1]
            MaxIdx.append(i)

    i += 1  # 跳两格

print(MaxIdx)

result = np.zeros(len_a)

for i in range(len(MaxIdx)):
    result[MaxIdx[i]] = a[MaxIdx[i]]

num_bins = len_a

plt.scatter(list(range(len_a)), a, marker='*')
plt.scatter(list(range(len(result))), result, marker='o')

plt.legend(['Ori_Data', 'Res_Data'], loc='best')

plt.show()
