import numpy as np
import kernel_create as krn


def canny_edge_detection(image):
    g_mask=krn.Gaussian_kernel(2,(13,13))
    horizental_mask=krn.Custom_kernel((1,2))
    vertical_mask=krn.Custom_kernel((2,1))


    filtered=krn.add_filter(image,g_mask)

    x_gradiant=krn.add_filter(filtered,horizental_mask)
    y_gradiant=krn.add_filter(filtered,vertical_mask)

    gradiant_magnitude=np.power(np.add(np.power(x_gradiant,2),np.power(y_gradiant,2)),0.5)

    theta=np.arctan2(y_gradiant,x_gradiant) * 180 / np.pi

    theta=(theta + 180)%180
    theta=((theta+22.5)//45)*45
    edges_detected=non_maxm(theta,gradiant_magnitude)
    strong_edges,weak_edges=two_thresholds(edges_detected)
    edges=hysteresis(gradiant_magnitude,strong_edges,weak_edges)

    return edges

def two_thresholds(magnitude:np.array):

    threshold2=0.2* magnitude.max()
    threshold1=0.1* magnitude.max()
    strong_edges=magnitude>threshold2
    weak_edges=np.logical_and(threshold1<magnitude,magnitude<threshold2)
    return strong_edges,weak_edges

def hysteresis(image,strong_edges,weak_edges):
    width=image.shape[1]
    height=image.shape[0]
    neighbors=([1,0],[1,1],[0,1],[-1,-1],[-1,0],[0,-1],[-1,+1],[1,-1])
    for i in range (height):
        for j in range(width):
            if weak_edges[i,j]==True:
                for neighbor in np.add([i,j],neighbors):
                    if (0<=neighbor[0]< height and 0<=neighbor[1]<width ) and  strong_edges[neighbor[0],neighbor[1]]==True:
                        image[i,j]=1
                        break
                if strong_edges[i,j]==False:
                    image[i,j]=0
    return image


def non_maxm(theta,gradiant_magnitude):

    width=theta.shape[1]
    height=theta.shape[0]

    edges=np.zeros(theta.shape)

    for i in range (1,height-1):
        for j in range(1,width-1):
            if theta[i,j]==90:

                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i,j+1],gradiant_magnitude[i,j-1])


            elif theta[i,j]==135:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i-1,j-1],gradiant_magnitude[i+1,j+1])


            elif theta[i,j]==0:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i-1,j],gradiant_magnitude[i+1,j])


            elif theta[i,j]==45:
                maximum=max(gradiant_magnitude[i,j],gradiant_magnitude[i-1,j+1],gradiant_magnitude[i+1,j-1])
                
            if  gradiant_magnitude[i,j]==maximum:
                edges[i,j]=gradiant_magnitude[i,j]


    return edges