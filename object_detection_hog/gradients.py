import numpy as np
from common.kernel import *

sobel_x = Custom_kernel("sobel_x")
sobel_y = Custom_kernel("sobel_y")


def compute_gradients(img):

    gx = add_filter(img, sobel_x)
    gy = add_filter(img, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    orientation = np.rad2deg(orientation) % 180

    return magnitude, orientation
