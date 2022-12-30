import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

im = cv2.imread('HW3/C1HW03-2022/Image 4-1.JPG') * 1.
im = (im[:, :, 0] + im[:, :, 1] + im[:, :,2]) / 3
def local_enhancement(img):
    im = img
    img_h, img_w = im.shape

    # 機率
    p = np.zeros(256)
    # print(p)
    for sample in range(256):
        count = 0
        for i in range(img_h):
            for j in range(img_w):
                if (im[i][j] == sample):
                    count += 1
        p[sample] = count / img_h / img_w
    # print(p)

    # 平均
    m = 0
    for i in range(256):
        m += i * p[i]
    # u2 
    u2 = 0
    for i in range(256):
        u2 += (i - m)**2 * p[i]
    # print(u2)
    # print(np.var(im))

    # 反射padding
    im_padded =  cv2.copyMakeBorder(im, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    # 計算區域的統計 --> 3x3
    # C = 4, k0 = 0.02, k1 = 0.4, k2 = 0.02, k3 = 0.4
    im_result = np.zeros(im.shape)
    C, k0, k1, k2, k3 = (100, 0, 0.25, 0, 0.1)
    for i in range(1, im_padded.shape[0] - 1):
        for j  in range(1, im_padded.shape[1] - 1):
            m_sy = np.average(im_padded[i - 1:i + 1, j - 1: j + 1])
            var_xy = np.var(im_padded[i - 1:i + 1, j - 1: j + 1])
            if ((k0 * m < m_sy and m_sy < k1 * m) and (k2 * u2 < var_xy and var_xy < k3 * u2)):
                im_result[i - 1][j - 1] = C * im[i - 1][j - 1]
                print(1)
            else:
                im_result[i - 1][j - 1] = im[i - 1][j - 1]
    return im_result


# histogram equalization
def cal_hist(matrix):
    hist = np.zeros(256)
    h, w = matrix.shape
    for i in range(256):
        count = 0
        for j in range(h):
            for k in range(w):
                if (i == matrix[j][k]):
                    count += 1
        hist[i] = count
    return hist
def cal_probability(img): # probability
    h, w = img.shape
    area  = h * w
    hist = cal_hist(img)
    p = hist / area
    return p
def cal_Cumulative_probability(p): # Cumulative_probability
    CP = np.zeros(len(p))
    count = 0
    for index, value in enumerate(p):
        count += value
        CP[index] = count * 255
    return CP
def equalization(img):
    CP = cal_Cumulative_probability(cal_probability(img))
    h, w = img.shape
    equalization_img = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            equalization_img[i][j] = CP[int(img[i][j])]
    return equalization_img

plt.imshow(np.uint8(im_result), cmap='gray')
# plt.imshow(equalization(im), cmap='gray')
plt.show()
# plt.bar(np.array(range(256)), p)
# plt.show()
    