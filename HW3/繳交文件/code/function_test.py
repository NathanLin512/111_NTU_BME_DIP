import cv2
import numpy as np
import math 
import time # calculate process time

from matplotlib import pyplot as plt # show subplot

# Read image
im = cv2 .imread('HW3/C1HW03-2022/Image 3-2.JPG')
# im = np.array([[[0, 255, 0]
#                 , [255, 255, 255]
#                 , [0, 255, 0]]
#                 ,[[0, 255, 0]
#                 , [255, 255, 255]
#                 , [0, 255, 0]],
#                 [[0, 255, 0]
#                 , [255, 255, 255]
#                 , [0, 255, 0]]])
# im = np.zeros((3, 3, 3))
# for i in range(3):
#     im[:, :, i] = np.array([[0, 255, 0]
#                             , [255, 255, 255]
#                             , [0, 255, 0]])
# im = np.array([[0, 255, 0]
#                 , [255, 255, 255]
#                 , [0, 255, 0]])
# print(im)

# BGR 轉換到 RGB (matplot show use)
def BGR2RGB(img):
    RGB = np.zeros(img.shape)
    RGB[:, :, 0] = img[:, :, 2]
    RGB[:, :, 1] = img[:, :, 1]
    RGB[:, :, 2] = img[:, :, 0]
    return RGB

# Padding (通用全部的mask)
def Padding(img, mask_size):
    img_h, img_w, img_k =img.shape
    mask_h, mask_w = mask_size
    pad_h = img_h + mask_h - 1
    pad_w = img_w + mask_w - 1
    result_padding = np.zeros((pad_h, pad_w, img_k))
    # 控制index範圍
    index_h, index_w = int((mask_h - 1) / 2), int((mask_w - 1) / 2)
    for k in range(img_k):
        for i in range(index_h, result_padding.shape[0] - index_h):
            for j in range(index_w, result_padding.shape[1] - index_w):
                result_padding[i][j][k] = img[i - index_h][j - index_w][k]
    return result_padding

# convolution (通用全部的mask)
def converlution(img_padding, mask):
    img_h, img_w, img_k = img_padding.shape
    mask_h, mask_w = mask.shape
    # 控制convolution 的圖片範圍
    conv_h, conv_w = int((mask_h - 1) / 2), int((mask_w - 1) / 2)
    conv = np.zeros((img_h - mask_h + 1, img_w - mask_w +1, img_k))
    # 控制index範圍
    index_h, index_w = int((mask_h - 1) / 2), int((mask_w - 1) / 2)
    for k in range(img_k):
        for i in range(index_h, img_padding.shape[0] - index_h):
            for j in range(index_w, img_padding.shape[1] - index_w):
                conv[i - index_h][j - index_w][k] = np.sum(mask * img_padding[i - conv_h : i + conv_h + 1, j - conv_w : j + conv_w + 1, k])
    return conv

# use Lowpass filters
# Generate average filters, size --> (3, 3) = 3x3; (5, 5) = 5x5 ...
def average_filters(size):
    h, w = size
    avg_f = np.ones((h, w)) / (h * w)
    return avg_f

# 一次運行均化濾波
def run_avg_filter(img, mask_size):
    start_time = time.time()
    img_padding = Padding(img, mask_size)
    avg_mask = average_filters(mask_size)
    conv = converlution(img_padding, avg_mask)
    process_time = time.time() - start_time
    return conv, process_time

# Low pass Gaussian
# Reference: https://medium.com/@bob800530/python-gaussian-filter-概念與實作-676aac52ea17
# Generate Gaussian filters
def Gaussian_filters(size, sigma):
    def Gaussian_op(x, y, sigma): # 建立運算fun
        result = 1 / (2 * math.pi * sigma**2) * math.exp(- (x**2 + y**2) / (2 * sigma**2))
        return result
    Gaussian = np.zeros(size)
    mask_h, mask_w = size
    for i in range(mask_h):
        for j in range(mask_w):
            Gaussian[i][j] = Gaussian_op(i - 1, j - 1, sigma)
    return Gaussian / np.sum(Gaussian)

# 一次運行高斯濾波
def run_Gaussian_filter(img, mask_size, sigma):
    start_time = time.time()
    img_padding = Padding(img, mask_size)
    # print(img_padding)
    Gaussian_mask = Gaussian_filters(mask_size, sigma)
    # print(Gaussian_mask)
    conv = converlution(img_padding, Gaussian_mask)
    process_time = time.time() - start_time
    # print(conv)
    return conv, process_time

# sobel op
# # run sobel x 
def run_sobel_x(img): # 目前只有3x3
    start_time = time.time()
    sobel_x = np.array([[1, 2, 1]
                        ,[0, 0, 0]
                        ,[-1, -2, -1]])
    img_padding = Padding(img, (3, 3))
    # conv = converlution(img_padding, sobel_x) # the function was made by myself
    conv = np.convolve(img_padding, sobel_x, 'valid') # use numpy function
    process_time = time.time() - start_time
    return conv, process_time


def edgesMarrHildreth1(img, sigma, mask_size):
    """
            finds the edges using MarrHildreth edge detection method...
            :param im : input image
            :param sigma : sigma is the std-deviation and refers to the spread of gaussian
            :return:
            a binary edge image...
    """
   
    # st.write(img.shape[0])
    # size = int(2*(np.ceil(3*sigma))+1)
    size = mask_size
    print(size)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    # st.write(x)
    # st.write(y)
    log = np.zeros_like(img, dtype=float)

    # applying filter O
    for k in range(img.shape[2]):
        for i in range(img.shape[0]-(kern_size-1)):
            for j in range(img.shape[1]-(kern_size-1)):
                window = img[i:i+kern_size, j:j+kern_size, k] * kernel
                log[i, j, k] = np.sum(window)
            


    log = log.astype(np.int64, copy=False)

    zero_crossing = np.zeros_like(log)

    # computing zero crossing O
    for k in range(log.shape[2]):
        for i in range(log.shape[0]-(kern_size-1)):
            for j in range(log.shape[1]-(kern_size-1)):
                if log[i][j][k] == 0:
                    if (log[i][j-1][k] < 0 and log[i][j+1][k] > 0) or (log[i][j-1][k] < 0 and log[i][j+1][k] < 0) or (log[i-1][j][k] < 0 and log[i+1][j][k] > 0) or (log[i-1][j][k] > 0 and log[i+1][j][k] < 0):
                        zero_crossing[i][j][k] = 255
                if log[i][j][k] < 0:
                    if (log[i][j-1][k] > 0) or (log[i][j+1][k] > 0) or (log[i-1][j][k] > 0) or (log[i+1][j][k] > 0):
                        zero_crossing[i][j][k] = 255


    return log, zero_crossing

def edgesMarrHildreth(img, sigma):
    """
            finds the edges using MarrHildreth edge detection method...
            :param im : input image
            :param sigma : sigma is the std-deviation and refers to the spread of gaussian
            :return:
            a binary edge image...
    """
   
    # st.write(img.shape[0])
    size = int(2*(np.ceil(3*sigma))+1)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    # st.write(x)
    # st.write(y)
    log = np.zeros_like(img, dtype=float)

    # applying filter O
    for i in range(img.shape[0]-(kern_size-1)):
        for j in range(img.shape[1]-(kern_size-1)):
            window = img[i:i+kern_size, j:j+kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)

    zero_crossing = np.zeros_like(log)

    # computing zero crossing O
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if log[i][j] == 0:
                if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
                    zero_crossing[i][j] = 255
size = (7, 7)
# # 均值濾波測試
# after_avg, process_time = run_avg_filter(im, (43, 43))
# print("Process time --> {:.2f} s".format(process_time))


# # 高斯濾波測試
# after_gauss, process_time = run_Gaussian_filter(im, size, 2.5)
# print("Process time --> {:.2f} s".format(process_time))

# after_sobelx, process_time = run_sobel_x(im)
# print("Process time --> {:.2f} s".format(process_time))

log, zero_crossing = edgesMarrHildreth(im, 4, 25)

cv2.imwrite('MH.jpg', zero_crossing)
# matplot show
origin_RGB = BGR2RGB(im)
after_RGB = BGR2RGB(log)
# after_RGB2 = BGR2RGB(zero_crossing)
# padding_RGB = BGR2RGB(Padding(im, size))
# print(im.shape)
# print(padding_RGB.shape)
plt.subplot(1, 3, 1)
plt.imshow(origin_RGB / 255)
plt.subplot(1, 3, 2)
plt.imshow(after_RGB / 255) 
# plt.subplot(1, 3, 3)
# plt.imshow(after_RGB2 / 255) 
plt.show()

# cv2.imshow('img',conv / 255)
# cv2.waitKey(0)

