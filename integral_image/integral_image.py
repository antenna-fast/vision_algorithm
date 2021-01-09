from numpy import *
img = ones((5, 6))  # lin5 row6

img_size = img.shape


# 初始化I
I = zeros(img_size)
I[0][0] = img[0][0]

# 
for i in range(1, img_size[0]):  # 首列
	I[i][0] = I[i-1][0] + img[i][0]

for i in range(img_size[1]):  # 首行	
	I[0][i] = I[0][i-1] + img[0][i] 

for i in range(1, img_size[0]):
	for j in range(1, img_size[1]):
		#print(i, j)
		I[i][j] = img[i][j] + I[i][j-1] + I[i-1][j] - I[i-1][j-1]

		print(I[i][j])
print(I)

# 计算任意矩形区域的面积
