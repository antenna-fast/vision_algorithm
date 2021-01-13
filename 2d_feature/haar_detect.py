from numpy import *
import matplotlib.pyplot as plt
import cv2
from integral_image import *

img = zeros((480, 640, 3)).astype(uint8)
# img = zeros((480, 640)).astype(uint)
# img[:240, :320] = array([200, 0, 0])
for i in range(100):
    for j in range(100):
        img[i][j] = array([0, 200, 0])
        # img[i][j] = 200

img = cv2.imread('images.jpg')

# 在灰度图上检测
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':

    img_size = img_g.shape

    # 积分图
    i_img = get_integral(img_g)

    # 执行haar detect
    # haar大小
    width = 3
    height = 8

    height_add = int(height/2)
    width_add = int(width/2)
    # 区域权重
    w_b, w_w = 1, -1  # 权重

    for i in range(img_size[0]-10):
        for j in range(img_size[1]-10):
            # 黑色窗口 像素值统计
            b_win = i_img[i+width][j+height_add] - i_img[i][j]
            # 白色窗口
            w_win = i_img[i+width][j+height] - i_img[i][j+height_add]
            res = w_b * b_win + w_w * w_win

            if res > 46:
                print(i, j)
                # print('res:', res)
                img[i+width_add][j+height_add] = [255, 0, 0]

    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    # plt.imshow(i_img, cmap='gray')
    # plt.imshow(img_g, cmap='gray')
    # plt.imshow(img)
    plt.show()