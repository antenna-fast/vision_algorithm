import matplotlib.pyplot as plt
from numpy import *

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }


# 输入一个序列  输出是否不平衡
def get_unbalance(vec, threshold, i):

    part_map = {6115: 'Ear', 2314: 'Tiptoe', 1937: 'Nose', 621: '', 8597: 'Belly',
                11223: 'Knee',
                # i: 'Test'
                }

    len_vec = len(vec)
    sort_vec = sort(vec)
    sort_vec_cut = sort_vec[:len_vec - 1]  # 前面的len-1个

    diff = sort_vec[1:] - sort_vec_cut  # 一个以后的-前面n-1个  包含n-1个元素  为了快速计算

    print('sort_vec:', sort_vec)
    print('diff:', diff)
    res = 0
    if max(diff) > threshold:
        res = 1

    # 可视化部分
    labels = list(range(len_vec))
    width = 0.08  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels,
           # vec,
           sort_vec,
           width,
           # yerr=men_std,  # 浮动
           # label='intensity'
           )

    # ax.set_ylabel('Scores', font1)
    ax.set_title(part_map[i], font1)
    # ax.legend(font1)

    y_stick = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0016]
    y_stick = list(arange(0, 0.0026, 0.0005))
    plt.yticks(y_stick)
    plt.show()

    return res


if __name__ == '__main__':
    i = 6115
    i = 8597
    # i = 11223
    a = loadtxt('../save_file/kl_buff_' + str(i) + '.txt')
    print(a)
    # a = array([3, 2, 4, 5, 1, 9])  # 序列
    res = get_unbalance(a, 1, i)
    print('res:', res)
