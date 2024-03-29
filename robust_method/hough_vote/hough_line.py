from numpy import *
import matplotlib.pyplot as plt


# 霍夫直线检测
# 基本思想练习.其他的只需要更换投票函数

if __name__ == '__main__':
    print('Hough line feature...')

    # gen_data
    # define y=2*x + 3
    x = np.arange(0, 10, 0.1)
    y = 2 * x + 3 + np.random.rand(len(x))  # ok


    fig = plt.figure()
    # print(dir(fig))
    ax = fig.add_subplot(111)
    ax.scatter(pts.T[0], pts.T[1])

    x_test = np.array([-1, 10])
    y_test = model_buff[idx][0] * x_test + model_buff[idx][1]
    ax.plot(x_test, y_test, color='r')
    plt.show()
