from numpy import *


# 通过对原始数据进行离散化采样，得到每个区间内的分布
if __name__ == '__main__':
    print()

    a = array(range(360))
    # print(a)

    sample = (a / 10).astype(int)  # 采样步长是10
    # print(sample)

    x_axis = unique(sample)  # 直方图的x轴
    # print(x_axis)

    num_space = []
    for x in x_axis:
        now_num = sum(where(sample == x, 1, 0))
        # print(now_num)
        num_space.append(now_num)

    print(num_space)
