import matplotlib.pyplot as plt
import numpy as np
import canny
import harris
import laplacian_pyramid
import cv2 as cv

image=cv.imread('lena_color.jpg',cv.IMREAD_GRAYSCALE)
image=image.astype(np.float32)/255.0
# image=canny.canny_edge_detection(image)
# image=harris.harris_corner_detection(image)
out=laplacian_pyramid.laplasian_pyramid(image)
plt.imshow(np.abs(out), cmap='gray')
# plt.imshow(image, cmap='gray')
plt.show()
