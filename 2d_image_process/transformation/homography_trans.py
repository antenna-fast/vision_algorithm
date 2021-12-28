import cv2
import numpy as np


if __name__ == '__main__':
    # read img
    img_path = ''
    img = cv2.imread(img_path)

    # def homography
    homo_mat = 0

    # trans
    img = cv2.transform(img, homo_mat)

    cv2.imshow('homo_trans', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
