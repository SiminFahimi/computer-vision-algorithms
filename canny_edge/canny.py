import numpy as np
import common.kernel  as krn


def canny_edge_detection(image):
    g_mask=krn.Gaussian_kernel(2,(13,13))

    # horizental_mask =krn.Custom_kernel("simple_x")
    # vertical_mask = krn.Custom_kernel("simple_y")

    horizental_mask = krn.Custom_kernel("sobel_x")
    vertical_mask   = krn.Custom_kernel("sobel_y")

    filtered=krn.add_filter(image,g_mask)

    x_gradiant=krn.add_filter(filtered,horizental_mask)
    y_gradiant=krn.add_filter(filtered,vertical_mask)

    gradient_magnitude =np.power(np.add(np.power(x_gradiant,2),np.power(y_gradiant,2)),0.5)
    theta=np.arctan2(y_gradiant,x_gradiant) * 180 / np.pi
    theta=(theta + 180)%180
    theta=((theta+22.5)//45)*45

    edges_detected=non_max_suppression(theta,gradient_magnitude)
    strong_edges,weak_edges=two_thresholds(edges_detected)
    edges=hysteresis(strong_edges,weak_edges)

    return (gradient_magnitude ,edges)

def two_thresholds(magnitude:np.array):

    threshold2=0.3* magnitude.max()
    threshold1=0.15* magnitude.max()
    strong_edges=magnitude>threshold2
    weak_edges=np.logical_and(threshold1<magnitude,magnitude<threshold2)
    return strong_edges,weak_edges

def hysteresis(strong_edges,weak_edges):
    height, width = strong_edges.shape

    result = np.zeros_like(strong_edges)

    neighbors = [(1,0),(1,1),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)]

    for i in range (1, height-1):
        for j in range(1, width-1):
            if weak_edges[i,j]:
                for dx, dy in neighbors:
                    ni, nj = i + dx, j + dy
                    if strong_edges[ni, nj]:
                        result[i, j] = 1
                        break
        
    result=np.logical_or(result,strong_edges)
    return result

def non_max_suppression(theta,gradiant_magnitude):

    height, width = theta.shape

    edges=np.zeros(theta.shape)

    for i in range (1,height-1):
        for j in range(1,width-1):

            if theta[i,j]==90:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i-1,j+1],gradiant_magnitude[i+1,j-1])

            elif theta[i,j]==135:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i-1,j-1],gradiant_magnitude[i+1,j+1])

            elif theta[i,j]==0:             
                maximum = max(gradiant_magnitude[i,j], gradiant_magnitude[i-1,j], gradiant_magnitude[i+1,j])

            elif theta[i,j]==45:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i,j+1],gradiant_magnitude[i,j-1])
                
            if  gradiant_magnitude[i,j]>= maximum:
                edges[i,j]=gradiant_magnitude[i,j]


    return edges