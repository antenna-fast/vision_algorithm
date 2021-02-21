"""
# 8th
编写一个程序完成如下功能：读入清晰图像，加上椒盐噪声
采用－－有选择保边缘平滑法－－对图像进行平滑。
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import random_noise

# 读入清晰图像
# einstein.tif
# lena512color.jpg
# mandril_color.jpg
# cameraman.tif
src = Image.open("einstein.tif").convert('L')  # 转换成灰度
src = np.array(src)

'''
加入椒盐噪声
输入：清晰灰度图
返回：噪声污染图
'''


def addnoise(img):
    print("addnoise is running ...")

    # 随机生成5000个椒盐点
    rows, cols = img.shape

    for i in range(5000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 255

    output = img
    return output


'''
平滑滤波器：有选择保边缘平滑法
    按照PPT 处理5*5邻域
输入：清晰灰度图\methord 为填充方案\可选zero or replicate
返回：滤波后的图像
'''


def filter(img, methord):
    # 核大小 5*5
    kernel_lenth = 5
    rect_kernel_lenth = 3  # 3*3 kernel size

    b = np.array(img)

    # 读取原始矩阵大小
    line_0, col_0 = b.shape
    print("Ori_size:", b.shape)
    # 创建输出画布 全零阵
    output = np.zeros((line_0 + 2 * (kernel_lenth - 1), col_0 + 2 * (kernel_lenth - 1)))
    # 填充边界
    if (methord == "zero"):
        # 填0 先填充行，加两行后填充列
        for i in range(kernel_lenth - 1):
            line, col = b.shape  # Updata Matrix size
            b = np.insert(b, 0, np.zeros(col), axis=0)
            b = np.insert(b, line + 1, np.zeros(col), axis=0)

            b = np.insert(b, 0, np.zeros(line + 2), axis=1)
            b = np.insert(b, col + 1, np.zeros(line + 2), axis=1)
        # 0填充完毕

    elif (methord == "replicate"):
        # 填边界值
        # 先行后列，这样不必专门处理角点
        for i in range(kernel_lenth - 1):  # kernel_lenth
            line, col = b.shape  # Updata Matrix size
            side_up = b[:1]
            side_down = b[line - 1:line]

            b = np.insert(b, 0, side_up, axis=0)
            b = np.insert(b, line + 1, side_down, axis=0)

            side_left = b[:, 0]  # 提取某一列
            side_right = b[:, col - 1]

            b = np.insert(b, 0, side_left, axis=1)
            b = np.insert(b, col + 1, side_right, axis=1)
    # 复制填充完毕
    # print("expsize:", b.shape) # to 5*5 Size OK!

    # 5*5填充后对于5边形和6边形是友善的
    line, col = b.shape  # Updata Matrix size

    # 3*3 矩形区域计算 为了简化逻辑，该部分与其它(5*5)区域分立
    for i in range(2, line - 2):  # kernel center x
        for j in range(2, col - 2):  # kernel center y slide
            # 对于每个像素建立2*1的均值方差表 每次清零
            var_list = [[], [], []]  # 0- 3;   5 6
            mean_list = [[], [], []]  # 0- 3;  5 6

            # 提取欲处理的<9>个子区域
            # 3*3
            mask_temp_3 = b[i - 1:i + 2, j - 1:j + 2]  # 3*3的就这样提取出来了  notice that 要到3*3  即0-2 ，后面的要取到3
            var_list[0].append(np.var(mask_temp_3))
            mean_list[0].append(np.mean(mask_temp_3))

            local_var1 = []
            local_mean1 = []
            local_var2 = []
            local_mean2 = []
            # 5*5
            mask_temp_5 = b[i - 2:i + 3, j - 2:j + 3]

            # 1
            list51 = [mask_temp_5[2][2], mask_temp_5[1][1], mask_temp_5[0][1], mask_temp_5[0][3], mask_temp_5[1][3],
                      mask_temp_5[1][2], mask_temp_5[0][2]]
            list52 = [mask_temp_5[2][2], mask_temp_5[1][3], mask_temp_5[1][4], mask_temp_5[2][4], mask_temp_5[3][4],
                      mask_temp_5[3][3], mask_temp_5[2][3]]
            list53 = [mask_temp_5[2][2], mask_temp_5[3][3], mask_temp_5[4][3], mask_temp_5[4][2], mask_temp_5[4][1],
                      mask_temp_5[3][1], mask_temp_5[3][2]]
            list54 = [mask_temp_5[2][2], mask_temp_5[3][1], mask_temp_5[3][0], mask_temp_5[2][0], mask_temp_5[1][0],
                      mask_temp_5[1][1], mask_temp_5[2][1]]

            local_var1.append(np.var(list51))
            local_var1.append(np.var(list52))
            local_var1.append(np.var(list53))
            local_var1.append(np.var(list54))
            #
            local_mean1.append(np.mean(list51))
            local_mean1.append(np.mean(list52))
            local_mean1.append(np.mean(list53))
            local_mean1.append(np.mean(list54))
            # 2
            list61 = [mask_temp_5[2][2], mask_temp_5[2][3], mask_temp_5[4][3], mask_temp_5[4][4], mask_temp_5[4][3],
                      mask_temp_5[3][2], mask_temp_5[4][4]]
            list62 = [mask_temp_5[2][2], mask_temp_5[3][2], mask_temp_5[4][1], mask_temp_5[4][0], mask_temp_5[3][0],
                      mask_temp_5[2][1], mask_temp_5[3][1]]
            list63 = [mask_temp_5[2][2], mask_temp_5[2][1], mask_temp_5[1][0], mask_temp_5[0][0], mask_temp_5[0][1],
                      mask_temp_5[1][2], mask_temp_5[1][1]]
            list64 = [mask_temp_5[2][2], mask_temp_5[1][2], mask_temp_5[0][3], mask_temp_5[0][4], mask_temp_5[1][4],
                      mask_temp_5[2][3], mask_temp_5[1][3]]
            local_var2.append(np.var(list61))
            local_var2.append(np.var(list62))
            local_var2.append(np.var(list63))
            local_var2.append(np.var(list64))

            local_mean2.append(np.mean(list61))
            local_mean2.append(np.mean(list62))
            local_mean2.append(np.mean(list63))
            local_mean2.append(np.mean(list64))

            min_var_temp = 0
            for m in range(4):  # 迭代查找5*5的最小方差 对应的均值
                if (local_var1[m] > local_var2[m]):
                    mean_value = local_mean2[m]
                    min_var_temp = local_var2[m]
                else:
                    mean_value = local_mean1[m]
                    min_var_temp = local_var1[m]
            # 找到包含3*3方差在内的最小方差位置
            # 根据索引找到对应的均值
            if var_list[0][0] > min_var_temp:  # 3的大 不取3
                output[i][j] = mean_value
            else:
                output[i][j] = mean_list[0][0]

            # output[i][j] = real_mean

        # print("Running...",i)

    # 裁剪
    output = output[kernel_lenth - 1:line - kernel_lenth + 1, kernel_lenth - 1:col - kernel_lenth + 1]
    # print("outputsize:",output.shape)
    return output


# Run to test
noise_img = random_noise(src, mode='s&p', amount=0.3)
filter_img = filter(noise_img, "zero")

filter_img2 = filter(noise_img, "replicate")

# print(filter_img)

src_dim1 = src.flatten()
noise_dim1 = noise_img.flatten()
filter_img_dim1 = filter_img.flatten()
filter_img_dim2 = filter_img2.flatten()

# 显示灰度图
plt.subplot(241), plt.imshow(src, 'gray'), plt.title('Src')
plt.axis('off')

plt.subplot(242), plt.imshow(noise_img, 'gray'), plt.title('NoiseImg')
plt.axis('off')

plt.subplot(243), plt.imshow(filter_img, 'gray'), plt.title('FilterImg_0')
plt.axis('off')

plt.subplot(244), plt.imshow(filter_img2, 'gray'), plt.title('FilterImg_copy')
plt.axis('off')

plt.subplot(245)
plt.title('SrcImg Hist')
n, bins, patches = plt.hist(src_dim1, bins=80, density=0, edgecolor='black', alpha=1, histtype='bar')

plt.subplot(246)
plt.title('NoiseImg Hist')
n, bins, patches = plt.hist(noise_dim1, bins=80, density=0, edgecolor='black', alpha=1, histtype='bar')

plt.subplot(247)
plt.title('FilterImg_0 Hist')
n, bins, patches = plt.hist(filter_img_dim1, bins=80, density=0, edgecolor='black', alpha=1, histtype='bar')

plt.subplot(248)
plt.title('FilterImg_copy Hist')
n, bins, patches = plt.hist(filter_img_dim2, bins=80, density=0, edgecolor='black', alpha=1, histtype='bar')

plt.show()

# Liu Yaohua - 2019.10.31 in UCAS
