from pyexpat import model
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from scipy import signal
from scipy import misc

im = cv2.imread('HW3/C1HW03-2022/Image 3-1.JPG') * 1.
im_gray = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) / 3

# padding and convolution --> https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def convolve2D(image, kernel, padding = 0, strides = 1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return imagePadded, output



# Generate average filters, size --> (3, 3) = 3x3; (5, 5) = 5x5 ...
def average_filters(size):
    h, w = size
    avg_f = np.ones((h, w)) / (h * w)
    return avg_f

# Generate gaussian filters
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

# BGR 轉換到 RGB (matplot show use)
def BGR2RGB(img):
    RGB = np.zeros(img.shape)
    print(RGB.shape)
    RGB[:, :, 0] = img[:, :, 2]
    RGB[:, :, 1] = img[:, :, 1]
    RGB[:, :, 2] = img[:, :, 0]
    return RGB


# run function
def image_process(img, kernel_name, kernel_size, sigma = 1):
    padding = int((kernel_size - 1) / 2)
    print(padding)
    kernel_size = (kernel_size, kernel_size)
    
    if (kernel_name == 'box'): # 均值濾波
        filters = average_filters(kernel_size)
    if (kernel_name == 'gaussian'):
        filters = Gaussian_filters(kernel_size, sigma)

    padded, result = convolve2D(img, filters, padding = padding)
    result2 = signal.convolve2d(padded, filters, mode='valid')
    return padded, result2

padded, result = image_process(im_gray, kernel_name = 'gaussian', kernel_size = 49, sigma = 5)

print(im_gray.shape)
print(padded.shape)
print(result.shape)
plt.subplot(1, 3, 1)
# plt.imshow(BGR2RGB(im / 255))
plt.imshow(im_gray, cmap = 'gray')
plt.subplot(1, 3, 2)
plt.imshow(padded, cmap = 'gray')
plt.subplot(1, 3, 3)
plt.imshow(result, cmap = 'gray')
plt.show()
