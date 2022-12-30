import argparse
import streamlit as st
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import sys
from PIL import Image
from skimage.color import rgb2gray ,rgba2rgb
from mpl_toolkits.mplot3d import Axes3D


st.title('Marr - Hildreth Edge Detection ')

def edgesMarrHildreth(img, sigma):
    """
            finds the edges using MarrHildreth edge detection method...
            :param im : input image
            :param sigma : sigma is the std-deviation and refers to the spread of gaussian
            :return:
            a binary edge image...
    """

    st.image(img ,use_column_width = True)
   
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

    # plotting images
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(log, cmap='gray')
    a.set_title('Laplacian of Gaussian')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(zero_crossing, cmap='gray')
    string = 'Zero Crossing sigma = '
    string += (str(sigma))
    a.set_title(string)
    # plt.show()

    st.pyplot(fig)

    return sigma*2, zero_crossing


def main():
    

    


    image = st.sidebar.file_uploader("Upload photo", type=(["png", "jpeg","jpg"]))

    for i in range(1, 6):
        st.sidebar.text("")
    
    x = st.sidebar.slider('Sigma' ,1 ,10)  
    

    if image is None:
        st.text("Please Upload Any Photo")
    else:
        
        img =io.imread(image)
        
        z =(img.shape)
        a =z[2]
        if a == 4:
            img = rgb2gray(rgba2rgb(img))
        elif a== 3:
            img = rgb2gray(img)
        else:
            img = img
            

        
        edgesMarrHildreth(img, x)


      
main()