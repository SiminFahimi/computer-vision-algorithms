import common.kernel as krl
import numpy as np
import math

def up_sample(image):
    h, w = image.shape
    up = np.zeros((h*2, w*2), dtype=image.dtype)
    up[::2, ::2] = image
    return up


def down_sample(image):
    return image[::2, ::2]


def laplacian_pyramid(image):

    h, w = image.shape

    gaussian_kernel = krl.Gaussian_kernel(1.0, (5,5))

    n = int(math.log2(min(h, w))) - 2
    n = max(1, n)

    gaussian = [None]*n
    laplacian = [None]*(n-1)
    recon = [None]*n

    gaussian[0] = image

    # build gaussian pyramid
    for i in range(1, n):
        smoothed = krl.add_filter(gaussian[i-1], gaussian_kernel)
        down = down_sample(smoothed)
        gaussian[i] = down

    # copy kernel for upsampling
    up_kernel = krl.Gaussian_kernel((0.9), (5,5))
    up_kernel.kernel *= 4

    # build laplacian pyramid
    for i in range(n-1):
        up = up_sample(gaussian[i+1])
        up = krl.add_filter(up, up_kernel)
        up = up[:gaussian[i].shape[0], :gaussian[i].shape[1]]
        laplacian[i] = gaussian[i] - up

    # reconstruction
    recon[n-1] = gaussian[n-1]

    for i in range(n-2, -1, -1):
        up = up_sample(recon[i+1])
        up = krl.add_filter(up, up_kernel)
        up = up[:laplacian[i].shape[0], :laplacian[i].shape[1]]
        recon[i] = up + laplacian[i]

    return gaussian, recon, laplacian
