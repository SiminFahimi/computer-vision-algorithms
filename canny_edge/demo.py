import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import canny

"""canny edge detection"""
def run_canny():

    image=cv.imread('lena_color.jpg',cv.IMREAD_GRAYSCALE)

    image=image.astype(np.float32)/255.0
    output=canny.canny_edge_detection(image)[1]
    plt.imshow(output, cmap='gray')
    plt.show()