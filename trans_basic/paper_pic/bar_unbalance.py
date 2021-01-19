import matplotlib.pyplot as plt
from numpy import *


# 输入一个序列  输出是否不平衡
def get_unbalance(vec, threshold):
    len_vec = len(vec)
    sort_vec = sort(vec)
    sort_vec_cut = sort_vec[:len_vec-1]  # 前面的len-1个

    diff = sort_vec[1:] - sort_vec_cut  # 一个以后的-前面n-1个  包含n-1个元素  为了快速计算

    print('sort_vec:', sort_vec)
    print('diff:', diff)
    res = 0
    if max(diff) > threshold:
        res = 1

    # 可视化部分
    labels = list(range(len_vec))
    width = 0.15  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels,
           # vec,
           sort_vec,
           width,
           # yerr=men_std,  # 浮动
           label='intensity')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.legend()

    plt.show()

    return res


if __name__ == '__main__':
       print()

       a = array([3, 2, 4, 5, 1, 9])  # 序列
       res = get_unbalance(a, 1)
       print('res:', res)
