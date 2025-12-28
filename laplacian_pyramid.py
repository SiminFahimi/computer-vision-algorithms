import kernel_create as krl
import numpy as np
import math

def up_sample(image):

    width,height=image.shape
    up=np.zeros((width*2,height*2))
    up[::2,::2]=image
    return up

def down_sample(image):

    down=image[::2,::2]   
    return down

def laplasian_pyramid(image):

    width,height=image.shape

    gaussian_kernel=krl.Gaussian_kernel(2,(13,13))
    n=int(math.log2(min(width,height))) -2

    g = [None]*n
    l = [None]*(n-1)
    x = [None]*n

    g[0]=image
    for _ in range(1,n):
        image=krl.add_filter(image,gaussian_kernel)
        image=down_sample(image)
        g[_]=image

    gaussian_kernel.kernel*=4
    for _ in range(0,n-1):
        up=up_sample(g[_+1])
        up=krl.add_filter(up,gaussian_kernel)
        up = up[:g[_].shape[0], :g[_].shape[1]]
        l[_] = g[_] - up
    
    x[n-1]=g[n-1]
    for _ in range(n-2, -1, -1):
        up = up_sample(x[_+1])
        up = krl.add_filter(up, gaussian_kernel)
        up = up[:l[_].shape[0], :l[_].shape[1]]
        x[_] = up + l[_]
    return l,x