"""harris corner detection"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from . import harris

def run_harris():
    image=cv.imread('data/square.jpg',cv.IMREAD_GRAYSCALE)
    image=image.astype(np.float32)/255.0
    output=harris.harris_corner_detection(image)
    plt.imshow(output, cmap='gray')
    plt.savefig("results/harris_corner_detection.png")
    plt.show()
    
run_harris()