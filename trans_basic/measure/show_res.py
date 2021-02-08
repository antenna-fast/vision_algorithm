from numpy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':

    x = list(range(5, 10))
    noise_rate = 0.1
    y = loadtxt('../save_file/repeat_buff_noise_' + str(noise_rate) + '.txt')

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(x, x, label='linear', marker='*')  # Plot some data on the axes.  # 第一条 线性
    ax.plot(x, y,
            # label='re',
            marker='o', color='darkorange')  # Plot more data on the axes...  # 二次
    # ax.plot(x, x**3, label='cubic', marker=10)  # ... and some more.  # 三次
    ax.set_xlabel('K Ring', font1)  # Add an x-label to the axes.
    ax.set_ylabel('Repeatability', font1)  # Add a y-label to the axes.
    # ax.set_title("Repeatability")  # Add a title to the axes.  # 标题

    for y in arange(0, 1.1, 0.1):
        plt.hlines(y, 5, 9, colors="gray", linestyles="dashed")

    for axis in [ax.xaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.show()
