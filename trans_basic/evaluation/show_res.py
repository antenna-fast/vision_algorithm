from numpy import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    x = array([0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.17])
    # y = array()
    y = loadtxt('../save_file/repeat_buff.txt')

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    # Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
    fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(x, x, label='linear', marker='*')  # Plot some data on the axes.  # 第一条 线性
    ax.plot(x, y, label='re', marker='o', color='darkorange')  # Plot more data on the axes...  # 二次
    # ax.plot(x, x**3, label='cubic', marker=10)  # ... and some more.  # 三次
    ax.set_xlabel('Noise', font1)  # Add an x-label to the axes.
    ax.set_ylabel('Repeatability', font1)  # Add a y-label to the axes.
    # ax.set_title("Repeatability")  # Add a title to the axes.  # 标题
    ax.legend()  # Add a legend.

    for y in arange(0, 1.1, 0.1):
        # y, xmin, xmax, colors=None, linestyles='solid', label=''
        plt.hlines(y, 0, 0.2, colors="gray", linestyles="dashed")

    plt.show()
