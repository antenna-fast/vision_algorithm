# 由轴角公式变换出

from numpy import *
from vec_pose import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    v = array([0, 0, 1])  # start
    k = array([1, 1, 1])

    # v_buff = [[0, 0, 0]]
    v_buff = []
    # v_buff = [[0, 0, 0], [1, 1.4, 1], [1, 0, 1], [3,3,3]]
    for t in range(1, 360, 12):
        # print('t:', t)
        v_rot = get_rodrigues(v, t, k)
        # print(v_rot)
        v_buff.append(v_rot)
    v_buff = array(v_buff).T

    print(v_buff)
    print(v_buff.shape)

    ax = plt.figure(1).gca(projection='3d')

    ax.plot_trisurf(v_buff[0], v_buff[1], v_buff[2], linewidth=0.2, antialiased=True, alpha=0.5)

    plt.scatter(v[0], v[1], v[2], marker='*')

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 11,
             }
    ax.set_xlabel("X Axis", font1)
    ax.set_ylabel("Y Axis", font1)
    ax.set_zlabel("Z Axis", font1)
    plt.show()
